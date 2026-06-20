"""Optimizer specs -- declarative until the compiler binds them to parameters."""

from __future__ import annotations

from typing import Iterable

import torch


class OptimizerSpec:
    def build(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:  # pragma: no cover
        raise NotImplementedError


class Adam(OptimizerSpec):
    def __init__(self, lr: float = 1e-3, **kwargs) -> None:
        self.lr = lr
        self.kwargs = kwargs

    def build(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.lr, **self.kwargs)


class SGD(OptimizerSpec):
    def __init__(self, lr: float = 1e-2, **kwargs) -> None:
        self.lr = lr
        self.kwargs = kwargs

    def build(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.SGD(params, lr=self.lr, **self.kwargs)
