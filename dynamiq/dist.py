"""Action distributions as graph nodes.

A distribution wraps a logits Signal (the network output) and exposes the
quantities objectives need -- ``log_prob`` of a taken action, ``entropy``,
per-action ``probs`` / ``log_probs`` -- each as a Signal carrying the right
provenance and gradient typing. Sampling for *acting* happens outside the graph
(in the env loop), using the live module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .signal import Context, Signal, combine_provenance


class _CatNode(Signal):
    def __init__(self, label: str, parents: list[Signal], provenance, carries_grad: bool) -> None:
        super().__init__(label=label, parents=parents, provenance=provenance, carries_grad=carries_grad)


class _CatLogProb(_CatNode):
    def eval(self, ctx: Context) -> torch.Tensor:
        logits = self.parents[0].eval(ctx)
        action = self.parents[1].eval(ctx).long()
        return torch.distributions.Categorical(logits=logits).log_prob(action)


class _CatEntropy(_CatNode):
    def eval(self, ctx: Context) -> torch.Tensor:
        logits = self.parents[0].eval(ctx)
        return torch.distributions.Categorical(logits=logits).entropy()


class _CatProbs(_CatNode):
    def eval(self, ctx: Context) -> torch.Tensor:
        return F.softmax(self.parents[0].eval(ctx), dim=-1)


class _CatLogProbsAll(_CatNode):
    def eval(self, ctx: Context) -> torch.Tensor:
        return F.log_softmax(self.parents[0].eval(ctx), dim=-1)


class Categorical:
    """A categorical policy over discrete actions, parameterized by logits."""

    def __init__(self, logits: Signal) -> None:
        self.logits = logits

    def log_prob(self, action: Signal) -> Signal:
        return _CatLogProb(
            label=f"logp({self.logits.label}, {action.label})",
            parents=[self.logits, action],
            provenance=combine_provenance(self.logits.provenance, action.provenance),
            carries_grad=self.logits.carries_grad,
        )

    def entropy(self) -> Signal:
        return _CatEntropy(
            label=f"H({self.logits.label})",
            parents=[self.logits],
            provenance=self.logits.provenance,
            carries_grad=self.logits.carries_grad,
        )

    def probs(self) -> Signal:
        return _CatProbs(
            label=f"probs({self.logits.label})",
            parents=[self.logits],
            provenance=self.logits.provenance,
            carries_grad=self.logits.carries_grad,
        )

    def log_probs_all(self) -> Signal:
        return _CatLogProbsAll(
            label=f"logp_all({self.logits.label})",
            parents=[self.logits],
            provenance=self.logits.provenance,
            carries_grad=self.logits.carries_grad,
        )


# -----------------------------------------------------------------------
# Squashed Normal (tanh-Gaussian) for continuous actions
# -----------------------------------------------------------------------

class _SquashedNode(Signal):
    def __init__(self, label: str, parents: list[Signal], provenance, carries_grad: bool) -> None:
        super().__init__(label=label, parents=parents, provenance=provenance, carries_grad=carries_grad)


class _SquashedLogProb(_SquashedNode):
    """log π(a|s) for a tanh-squashed Gaussian, with the Jacobian correction."""

    def __init__(self, label, parents, provenance, carries_grad, action_dim):
        super().__init__(label, parents, provenance, carries_grad)
        self.action_dim = action_dim

    def eval(self, ctx: Context) -> torch.Tensor:
        params = self.parents[0].eval(ctx)
        action = self.parents[1].eval(ctx)
        mean, log_std = params.chunk(2, dim=-1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        # atanh to recover the pre-squash sample
        u = torch.atanh(action.clamp(-0.999999, 0.999999))
        log_prob = dist.log_prob(u).sum(-1)
        # Jacobian correction: -sum log(1 - tanh^2(u))
        log_prob = log_prob - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return log_prob


class _SquashedRsample(_SquashedNode):
    """Reparameterized sample from tanh(Normal(mean, std))."""

    def __init__(self, label, parents, provenance, carries_grad, action_dim):
        super().__init__(label, parents, provenance, carries_grad)
        self.action_dim = action_dim

    def eval(self, ctx: Context) -> torch.Tensor:
        params = self.parents[0].eval(ctx)
        mean, log_std = params.chunk(2, dim=-1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return torch.tanh(dist.rsample())


class _SquashedEntropy(_SquashedNode):
    """Approximate entropy of the squashed Gaussian (Gaussian entropy, ignoring tanh)."""

    def __init__(self, label, parents, provenance, carries_grad, action_dim):
        super().__init__(label, parents, provenance, carries_grad)
        self.action_dim = action_dim

    def eval(self, ctx: Context) -> torch.Tensor:
        params = self.parents[0].eval(ctx)
        _, log_std = params.chunk(2, dim=-1)
        return (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum(-1)


class SquashedNormal:
    """A tanh-squashed Gaussian policy for continuous actions.

    Constructed from the output of a ``GaussianPolicy`` network, which produces
    ``(mean, log_std)`` concatenated along the last dimension.
    """

    def __init__(self, params: Signal, action_dim: int) -> None:
        self.params = params
        self.action_dim = action_dim

    def log_prob(self, action: Signal) -> Signal:
        return _SquashedLogProb(
            label=f"squashed_logp({self.params.label})",
            parents=[self.params, action],
            provenance=combine_provenance(self.params.provenance, action.provenance),
            carries_grad=self.params.carries_grad,
            action_dim=self.action_dim,
        )

    def rsample(self) -> Signal:
        return _SquashedRsample(
            label=f"squashed_rsample({self.params.label})",
            parents=[self.params],
            provenance=self.params.provenance,
            carries_grad=self.params.carries_grad,
            action_dim=self.action_dim,
        )

    def entropy(self) -> Signal:
        return _SquashedEntropy(
            label=f"squashed_H({self.params.label})",
            parents=[self.params],
            provenance=self.params.provenance,
            carries_grad=self.params.carries_grad,
            action_dim=self.action_dim,
        )
