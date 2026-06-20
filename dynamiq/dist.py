"""Action distributions as graph nodes.

A distribution wraps a logits Signal (the network output) and exposes the
quantities objectives need -- ``log_prob`` of a taken action, ``entropy``,
per-action ``probs`` / ``log_probs`` -- each as a Signal carrying the right
provenance and gradient typing. Sampling for *acting* happens outside the graph
(in the env loop), using the live module.
"""

from __future__ import annotations

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
