"""Learnable scalar parameters as graph nodes (e.g. SAC's temperature).

A :class:`Parameter` is a leaf node holding a single trainable scalar. The
compiler discovers it, instantiates it, and folds it into the optimizer just
like a network's weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .signal import Context, Provenance, Signal


class _ScalarModule(nn.Module):
    def __init__(self, init: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init)))

    def forward(self) -> torch.Tensor:
        return self.value


class Parameter(Signal):
    """A single learnable scalar (no data dependence, always differentiable)."""

    def __init__(self, name: str, init: float = 0.0) -> None:
        super().__init__(
            label=name, parents=[], provenance=Provenance.AGNOSTIC, carries_grad=True
        )
        self.name = name
        self._init = init
        self.module: nn.Module | None = None

    def instantiate(self) -> nn.Module:
        if self.module is None:
            self.module = _ScalarModule(self._init)
        return self.module

    def eval(self, ctx: Context) -> torch.Tensor:
        assert self.module is not None
        return self.module()
