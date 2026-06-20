"""Learnable function approximators as graph nodes.

A ``Network`` is a declarative handle to a parameterized function. You call it
on a Signal to get a Signal back -- e.g. ``q(batch.obs)``. The actual
``nn.Module`` is instantiated lazily by the compiler, so the graph stays a pure
description until you compile it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .signal import Context, Provenance, Signal


class NetworkApply(Signal):
    """The signal produced by applying a Network to an input signal."""

    def __init__(self, net: "Network", x: Signal) -> None:
        super().__init__(
            label=f"{net.name}({x.label})",
            parents=[x],
            # Output depends on the network's learnable params, so it carries
            # gradients regardless of whether the input did.
            provenance=x.provenance,
            carries_grad=True,
        )
        self.net = net

    def eval(self, ctx: Context) -> torch.Tensor:
        return self.net.module(self.parents[0].eval(ctx))


class Network:
    """Declarative learnable function. Subclasses provide a ``build`` method that
    returns the concrete ``nn.Module``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.module: nn.Module | None = None  # set by the compiler

    def build(self) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def instantiate(self) -> nn.Module:
        if self.module is None:
            self.module = self.build()
        return self.module

    def __call__(self, x: Signal) -> Signal:
        return NetworkApply(self, x)

    def __repr__(self) -> str:
        return f"<Network {self.name}>"


def _mlp(sizes: list[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class QNetwork(Network):
    """A state-action value head: obs -> one value per discrete action."""

    def __init__(
        self,
        name: str,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(name)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.activation = activation

    def build(self) -> nn.Module:
        return _mlp([self.obs_dim, *self.hidden, self.n_actions], self.activation)


class PolicyNetwork(Network):
    """A categorical policy head: obs -> logits over discrete actions.

    Provided so the type system has an on-policy consumer to reason about; full
    PPO/A2C training lands in Slice 1.
    """

    def __init__(
        self,
        name: str,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (128, 128),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__(name)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.activation = activation

    def build(self) -> nn.Module:
        return _mlp([self.obs_dim, *self.hidden, self.n_actions], self.activation)
