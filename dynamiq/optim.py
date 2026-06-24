"""Optimizer and LR scheduler specs -- declarative until the compiler binds them."""

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


# --------------------------------------------------------------------------
# Learning rate schedulers
# --------------------------------------------------------------------------

class SchedulerSpec:
    """Base class for declarative LR scheduler specs."""

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:  # pragma: no cover
        raise NotImplementedError


class CosineAnnealing(SchedulerSpec):
    """Cosine annealing schedule: LR decays from initial to ``eta_min`` over
    ``T_max`` steps, then optionally restarts.

    Wraps ``torch.optim.lr_scheduler.CosineAnnealingLR``.
    """

    def __init__(self, T_max: int, eta_min: float = 0.0) -> None:
        self.T_max = T_max
        self.eta_min = eta_min

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.eta_min,
        )


class LinearDecay(SchedulerSpec):
    """Linear LR decay from initial LR to ``end_factor * initial_lr`` over
    ``total_steps``.

    Wraps ``torch.optim.lr_scheduler.LinearLR``.
    """

    def __init__(self, total_steps: int, end_factor: float = 0.0) -> None:
        self.total_steps = total_steps
        self.end_factor = end_factor

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=self.end_factor,
            total_iters=self.total_steps,
        )


class StepDecay(SchedulerSpec):
    """Step LR decay: multiply LR by ``gamma`` every ``step_size`` steps.

    Wraps ``torch.optim.lr_scheduler.StepLR``.
    """

    def __init__(self, step_size: int, gamma: float = 0.1) -> None:
        self.step_size = step_size
        self.gamma = gamma

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma,
        )
