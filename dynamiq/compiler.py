"""The compiler: graph -> type-checked, executable training step.

``compile`` is where the paradigm pays off. It:
  1. collects every node in the objective graph(s),
  2. runs the RL type checker (the bit that turns 3-AM bugs into loud errors),
  3. instantiates networks / targets / parameters / optimizer,
  4. returns an :class:`Algorithm` whose ``step()`` runs one optimized update
     minimizing the weighted sum of objectives.
"""

from __future__ import annotations

import torch

from .exceptions import DynamiqTypeError
from .losses import Detached, Loss, RequiresProv
from .networks import Network, NetworkApply
from .optim import OptimizerSpec
from .params import Parameter
from .signal import DataField, Provenance, Signal
from .sources import Source
from .targets import Target, TargetApply


def _collect(losses: list[Loss]) -> list[Signal]:
    seen: dict[int, Signal] = {}
    order: list[Signal] = []
    for loss in losses:
        for node in loss.walk():
            if id(node) not in seen:
                seen[id(node)] = node
                order.append(node)
    return order


def _type_check(losses: list[Loss], nodes: list[Signal]) -> None:
    """Static RL type checking. Raises DynamiqTypeError on the first violation,
    with a message that names the offending node and how to fix it."""

    # 1. Mixing on-policy and off-policy data anywhere is illegal.
    for node in nodes:
        if node.provenance is Provenance.CONFLICT:
            raise DynamiqTypeError(
                f"Signal `{node.label}` mixes on-policy and off-policy data. "
                "An update rule must be derived from a single data distribution; "
                "split it or correct one source's provenance."
            )

    # 2. Per-objective obligations (referencing specific nodes).
    for loss in losses:
        for ob in loss.obligations:
            if isinstance(ob, Detached):
                if ob.node.carries_grad:
                    raise DynamiqTypeError(
                        f"Objective `{loss.label}` requires `{ob.node.label}` to be "
                        "a stop-gradient, but it still carries gradients. A "
                        "bootstrap/advantage/old-policy target must be detached. "
                        "Fix: build it from `dq.Target(...)`, use rollout data, or "
                        "call `.detach()`."
                    )
            elif isinstance(ob, RequiresProv):
                if ob.node.provenance is not ob.required:
                    raise DynamiqTypeError(
                        f"Objective `{loss.label}` requires {ob.required} data but "
                        f"`{ob.node.label}` is {ob.node.provenance}. (e.g. on-policy "
                        "objectives cannot train on replay-buffer samples.)"
                    )


class Algorithm:
    """A compiled, runnable algorithm: modules, optimizer, and the type-checked
    objective graph. ``step()`` performs one weighted-sum update."""

    def __init__(self, losses, parametrics, targets, sources, optimizer, device,
                 max_grad_norm=None, max_grad_value=None) -> None:
        self.losses = losses
        # parametrics: name -> object exposing `.module` (Network or Parameter)
        self.parametrics = {p.name: p for p in parametrics}
        self.targets = targets
        self.sources = sources
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        self._learnable = [
            p for pm in parametrics for p in pm.module.parameters() if p.requires_grad
        ]
        self.update_step = 0

    def module(self, name: str) -> torch.nn.Module:
        """Return the live ``nn.Module`` for a named network/parameter (to act with)."""
        return self.parametrics[name].module

    def step(self) -> dict[str, float]:
        ctx: dict[str, torch.Tensor] = {}
        for src in self.sources:
            ctx.update(src.materialize(self.device))

        terms = [loss.weight * loss.eval(ctx) for loss in self.losses]
        total = terms[0]
        for t in terms[1:]:
            total = total + t

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._learnable, self.max_grad_norm)
        if self.max_grad_value is not None:
            torch.nn.utils.clip_grad_value_(self._learnable, self.max_grad_value)
        self.optimizer.step()
        self.update_step += 1

        for tgt in self.targets:
            tgt.maybe_sync(self.update_step)

        metrics = {"loss": float(total.detach())}
        for i, term in enumerate(terms):
            metrics[f"loss/{self.losses[i].label[:22]}"] = float(term.detach())
        return metrics


def compile(objective, opt: OptimizerSpec, device=None,
            max_grad_norm: float | None = None,
            max_grad_value: float | None = None) -> Algorithm:
    """Type-check an objective graph and build a runnable :class:`Algorithm`.

    Parameters
    ----------
    max_grad_norm:
        If set, clip gradients by global L2 norm before each optimizer step.
    max_grad_value:
        If set, clamp each gradient element to [-value, +value] before each step.
    """
    losses = objective if isinstance(objective, list) else [objective]
    for o in losses:
        if not isinstance(o, Loss):
            raise DynamiqTypeError(
                f"compile() expects Loss objective(s); got {type(o).__name__}."
            )

    nodes = _collect(losses)
    _type_check(losses, nodes)  # <-- fails loudly here, before any training

    # Discover networks, targets, learnable parameters, and data sources.
    networks: dict[int, Network] = {}
    targets: dict[int, Target] = {}
    params: dict[int, Parameter] = {}
    sources: dict[int, Source] = {}
    for node in nodes:
        if isinstance(node, NetworkApply):
            networks[id(node.net)] = node.net
        elif isinstance(node, TargetApply):
            targets[id(node.target)] = node.target
            networks[id(node.target.source)] = node.target.source
        elif isinstance(node, Parameter):
            params[id(node)] = node
        elif isinstance(node, DataField):
            src = getattr(node, "source", None)
            if src is not None:
                sources[id(src)] = src

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Instantiate networks first (targets clone from them), then params.
    for net in networks.values():
        net.instantiate().to(device)
    for tgt in targets.values():
        tgt.instantiate().to(device)
    for p in params.values():
        p.instantiate().to(device)

    parametrics = list(networks.values()) + list(params.values())
    learnable = [
        prm
        for pm in parametrics
        for prm in pm.module.parameters()
        if prm.requires_grad
    ]
    optimizer = opt.build(learnable)

    return Algorithm(
        losses=losses,
        parametrics=parametrics,
        targets=list(targets.values()),
        sources=list(sources.values()),
        optimizer=optimizer,
        device=device,
        max_grad_norm=max_grad_norm,
        max_grad_value=max_grad_value,
    )
