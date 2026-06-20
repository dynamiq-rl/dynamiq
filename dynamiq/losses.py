"""Objectives -- the leaves of an algorithm graph the compiler minimizes.

An objective records its *typing obligations* as a list referencing specific
graph nodes, so the compiler can check them statically no matter how the
objective was composed:

  * ``Detached(node)``        -- ``node`` must not carry gradients (stop-grad).
  * ``RequiresProv(node, p)`` -- ``node``'s data must have provenance ``p``.

Objectives carry a ``weight`` and may be combined; ``dq.compile`` minimizes the
weighted sum. The generic :func:`minimize` objective turns *any* scalar signal
into a trainable term, which is what lets SAC's actor/temperature losses be
expressed without bespoke machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .signal import Context, Provenance, Signal, as_signal, combine_provenance


@dataclass
class Detached:
    """Obligation: the referenced node must be a stop-gradient."""

    node: Signal


@dataclass
class RequiresProv:
    """Obligation: the referenced node's data must have this provenance."""

    node: Signal
    required: Provenance


class Loss(Signal):
    def __init__(
        self,
        label: str,
        parents: list[Signal],
        reduce_fn,
        obligations: list,
        weight: float = 1.0,
    ) -> None:
        prov = Provenance.AGNOSTIC
        for p in parents:
            prov = combine_provenance(prov, p.provenance)
        super().__init__(label=label, parents=parents, provenance=prov, carries_grad=True)
        self.reduce_fn = reduce_fn
        self.obligations = obligations
        self.weight = weight

    def eval(self, ctx: Context) -> torch.Tensor:
        return self.reduce_fn([p.eval(ctx) for p in self.parents])

    def scale(self, w: float) -> "Loss":
        """Return the same objective with its weight multiplied by ``w``."""
        return Loss(self.label, self.parents, self.reduce_fn, self.obligations, self.weight * w)


# --------------------------------------------------------------------------
# Generic objective
# --------------------------------------------------------------------------
def minimize(
    scalar: Signal,
    *,
    detach: tuple[Signal, ...] = (),
    requires_provenance: Provenance | None = None,
    weight: float = 1.0,
) -> Loss:
    """Minimize an arbitrary scalar signal.

    ``detach`` lists nodes that must be stop-gradients (checked statically);
    ``requires_provenance`` constrains the data distribution feeding the term.
    """
    scalar = as_signal(scalar)
    obligations: list = [Detached(n) for n in detach]
    if requires_provenance is not None:
        obligations.append(RequiresProv(scalar, requires_provenance))
    return Loss(
        label=f"minimize({scalar.label})",
        parents=[scalar],
        reduce_fn=lambda xs: xs[0].mean() if xs[0].ndim > 0 else xs[0],
        obligations=obligations,
        weight=weight,
    )


def maximize(scalar: Signal, **kw) -> Loss:
    """Maximize a scalar signal (entropy bonuses, etc.)."""
    return minimize(-as_signal(scalar), **kw)


# --------------------------------------------------------------------------
# Regression objectives (value/critic learning)
# --------------------------------------------------------------------------
def huber(prediction: Signal, target: Signal, delta: float = 1.0, weight: float = 1.0) -> Loss:
    """Smooth-L1 regression toward a detached target."""
    prediction, target = as_signal(prediction), as_signal(target)
    return Loss(
        label=f"huber({prediction.label}, {target.label})",
        parents=[prediction, target],
        reduce_fn=lambda xs: F.smooth_l1_loss(xs[0], xs[1], beta=delta),
        obligations=[Detached(target)],
        weight=weight,
    )


def mse(prediction: Signal, target: Signal, weight: float = 1.0) -> Loss:
    """Mean-squared regression toward a detached target."""
    prediction, target = as_signal(prediction), as_signal(target)
    return Loss(
        label=f"mse({prediction.label}, {target.label})",
        parents=[prediction, target],
        reduce_fn=lambda xs: F.mse_loss(xs[0], xs[1]),
        obligations=[Detached(target)],
        weight=weight,
    )


# --------------------------------------------------------------------------
# Policy-gradient objectives
# --------------------------------------------------------------------------
def policy_gradient(log_prob: Signal, advantage: Signal, weight: float = 1.0) -> Loss:
    """REINFORCE/A2C: ``-(log_prob * advantage).mean()``. On-policy; advantage detached."""
    log_prob, advantage = as_signal(log_prob), as_signal(advantage)
    return Loss(
        label=f"pg({log_prob.label}, {advantage.label})",
        parents=[log_prob, advantage],
        reduce_fn=lambda xs: -(xs[0] * xs[1]).mean(),
        obligations=[Detached(advantage), RequiresProv(log_prob, Provenance.ON_POLICY)],
        weight=weight,
    )


def ppo_clip(
    new_log_prob: Signal,
    old_log_prob: Signal,
    advantage: Signal,
    clip: float = 0.2,
    weight: float = 1.0,
) -> Loss:
    """PPO clipped surrogate objective.

    Built from primitives: ``ratio = exp(new - old)``, then
    ``-min(ratio*A, clip(ratio)*A)``. Requires on-policy data; ``old_log_prob``
    and ``advantage`` must be stop-gradients.
    """
    new_log_prob = as_signal(new_log_prob)
    old_log_prob = as_signal(old_log_prob)
    advantage = as_signal(advantage)

    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage
    from .signal import minimum

    surrogate = minimum(unclipped, clipped)
    return Loss(
        label=f"ppo_clip({new_log_prob.label})",
        parents=[surrogate],
        reduce_fn=lambda xs: -xs[0].mean(),
        obligations=[
            Detached(old_log_prob),
            Detached(advantage),
            RequiresProv(new_log_prob, Provenance.ON_POLICY),
        ],
        weight=weight,
    )
