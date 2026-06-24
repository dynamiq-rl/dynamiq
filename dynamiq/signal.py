"""The Dynamiq IR.

An RL algorithm in Dynamiq is *not* a training loop you write by hand. It is a
graph of typed ``Signal`` nodes. You build the graph with ordinary Python
operators; the compiler walks it, type-checks it, and turns it into the loop.

Every Signal carries two pieces of static type information that ordinary tensors
do not:

  * ``provenance`` -- where the data came from (on-policy / off-policy /
    agnostic). This lets the compiler reject e.g. off-policy data fed into an
    on-policy objective.
  * ``carries_grad`` -- whether evaluating this node produces a tensor that is
    differentiably connected to learnable parameters. This lets the compiler
    reject bootstrap targets that were never detached -- the single most common
    silent RL bug.

These are computed structurally as the graph is built, so they cost nothing at
runtime and are available for static analysis (``verify`` / ``explain``).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:  # pragma: no cover
    from .networks import Network


class Provenance(enum.Enum):
    """Static label describing the distribution a signal's data was drawn from."""

    AGNOSTIC = "agnostic"      # constants / network params -- no data dependence
    ON_POLICY = "on_policy"    # collected from the current policy (fresh rollouts)
    OFF_POLICY = "off_policy"  # sampled from a replay buffer / arbitrary behaviour
    CONFLICT = "conflict"      # on-policy and off-policy data were combined (illegal)

    def __str__(self) -> str:
        return self.value


def combine_provenance(a: Provenance, b: Provenance) -> Provenance:
    """Provenance algebra for binary ops.

    AGNOSTIC is the identity element. Mixing ON_POLICY with OFF_POLICY is a
    hard error and yields CONFLICT, which the compiler surfaces with context.
    """
    if a is Provenance.CONFLICT or b is Provenance.CONFLICT:
        return Provenance.CONFLICT
    if a is Provenance.AGNOSTIC:
        return b
    if b is Provenance.AGNOSTIC:
        return a
    if a is b:
        return a
    return Provenance.CONFLICT


# A runtime context is just the materialized batch for this step: field -> tensor.
Context = dict[str, torch.Tensor]


class Signal:
    """A node in the algorithm graph.

    Subclasses implement :meth:`eval`, which computes the concrete tensor given
    a materialized batch. Users almost never construct subclasses directly --
    they fall out of operators (``+``, ``*``, indexing, ``.max()`` ...) and of
    calling networks / losses.
    """

    def __init__(
        self,
        label: str,
        parents: list["Signal"],
        provenance: Provenance,
        carries_grad: bool,
    ) -> None:
        self.label = label
        self.parents = parents
        self.provenance = provenance
        self.carries_grad = carries_grad

    # --- evaluation -------------------------------------------------------
    def eval(self, ctx: Context) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- graph traversal --------------------------------------------------
    def walk(self) -> "list[Signal]":
        """Return all unique nodes in this signal's subgraph (post-order)."""
        seen: dict[int, Signal] = {}
        order: list[Signal] = []

        def visit(node: Signal) -> None:
            if id(node) in seen:
                return
            seen[id(node)] = node
            for p in node.parents:
                visit(p)
            order.append(node)

        visit(self)
        return order

    # --- operator overloads ----------------------------------------------
    # Each returns a new Signal, propagating provenance / carries_grad.
    def _binary(self, other: Any, op: Callable, symbol: str) -> "Signal":
        other_sig = as_signal(other)
        return BinaryOp(self, other_sig, op, symbol)

    def __add__(self, other: Any) -> "Signal":
        return self._binary(other, torch.add, "+")

    def __radd__(self, other: Any) -> "Signal":
        return as_signal(other)._binary(self, torch.add, "+")

    def __sub__(self, other: Any) -> "Signal":
        return self._binary(other, torch.subtract, "-")

    def __rsub__(self, other: Any) -> "Signal":
        return as_signal(other)._binary(self, torch.subtract, "-")

    def __mul__(self, other: Any) -> "Signal":
        return self._binary(other, torch.multiply, "*")

    def __rmul__(self, other: Any) -> "Signal":
        return as_signal(other)._binary(self, torch.multiply, "*")

    def __truediv__(self, other: Any) -> "Signal":
        return self._binary(other, torch.divide, "/")

    def __neg__(self) -> "Signal":
        return UnaryOp(self, torch.neg, "-", keep_grad=True)

    def __getitem__(self, index: Any) -> "Signal":
        # q(obs)[actions] -> gather the value of the taken action along last dim.
        index_sig = as_signal(index)
        return GatherAction(self, index_sig)

    # --- reductions / elementwise ----------------------------------------
    def max(self, dim: int = -1) -> "Signal":
        return Reduce(self, "max", dim)

    def min(self, dim: int = -1) -> "Signal":
        return Reduce(self, "min", dim)

    def mean(self, dim: int | None = None) -> "Signal":
        return Reduce(self, "mean", dim)

    def sum(self, dim: int | None = None) -> "Signal":
        return Reduce(self, "sum", dim)

    def log(self) -> "Signal":
        return UnaryOp(self, torch.log, "log", keep_grad=True)

    def exp(self) -> "Signal":
        return UnaryOp(self, torch.exp, "exp", keep_grad=True)

    def detach(self) -> "Signal":
        """Explicit stop-gradient. Marks the subgraph result as non-differentiable."""
        return UnaryOp(self, lambda t: t.detach(), "detach", keep_grad=False)

    def clamp(self, lo: float, hi: float) -> "Signal":
        return UnaryOp(self, lambda t: t.clamp(lo, hi), f"clamp[{lo},{hi}]", keep_grad=True)

    def squeeze(self, dim: int = -1) -> "Signal":
        return UnaryOp(self, lambda t: t.squeeze(dim), f"squeeze({dim})", keep_grad=True)

    def __repr__(self) -> str:
        g = "grad" if self.carries_grad else "stop"
        return f"<Signal {self.label} [{self.provenance}, {g}]>"


# --------------------------------------------------------------------------
# Concrete node types
# --------------------------------------------------------------------------
class Constant(Signal):
    """A literal scalar / tensor with no data or parameter dependence."""

    def __init__(self, value: Any) -> None:
        super().__init__(
            label=f"const({value})",
            parents=[],
            provenance=Provenance.AGNOSTIC,
            carries_grad=False,
        )
        self.value = value

    def eval(self, ctx: Context) -> torch.Tensor:
        if isinstance(self.value, torch.Tensor):
            return self.value
        return torch.as_tensor(self.value)


def as_signal(x: Any) -> Signal:
    """Coerce a Python scalar / tensor into a Signal (Constant); pass Signals through."""
    if isinstance(x, Signal):
        return x
    return Constant(x)


def minimum(a: Any, b: Any) -> Signal:
    """Elementwise minimum of two signals (used by e.g. PPO clip, twin-Q SAC)."""
    a, b = as_signal(a), as_signal(b)
    return BinaryOp(a, b, torch.minimum, "min2")


def maximum(a: Any, b: Any) -> Signal:
    """Elementwise maximum of two signals."""
    a, b = as_signal(a), as_signal(b)
    return BinaryOp(a, b, torch.maximum, "max2")


class BinaryOp(Signal):
    def __init__(self, a: Signal, b: Signal, op: Callable, symbol: str) -> None:
        super().__init__(
            label=f"({a.label} {symbol} {b.label})",
            parents=[a, b],
            provenance=combine_provenance(a.provenance, b.provenance),
            carries_grad=a.carries_grad or b.carries_grad,
        )
        self.op = op

    def eval(self, ctx: Context) -> torch.Tensor:
        return self.op(self.parents[0].eval(ctx), self.parents[1].eval(ctx))


class UnaryOp(Signal):
    def __init__(self, a: Signal, op: Callable, name: str, keep_grad: bool) -> None:
        super().__init__(
            label=f"{name}({a.label})",
            parents=[a],
            provenance=a.provenance,
            carries_grad=a.carries_grad and keep_grad,
        )
        self.op = op

    def eval(self, ctx: Context) -> torch.Tensor:
        return self.op(self.parents[0].eval(ctx))


class Reduce(Signal):
    def __init__(self, a: Signal, kind: str, dim: int | None) -> None:
        super().__init__(
            label=f"{kind}({a.label}, dim={dim})",
            parents=[a],
            provenance=a.provenance,
            carries_grad=a.carries_grad,
        )
        self.kind = kind
        self.dim = dim

    def eval(self, ctx: Context) -> torch.Tensor:
        t = self.parents[0].eval(ctx)
        if self.kind == "max":
            return t.max(dim=self.dim).values if self.dim is not None else t.max()
        if self.kind == "min":
            return t.min(dim=self.dim).values if self.dim is not None else t.min()
        if self.kind == "mean":
            return t.mean(dim=self.dim) if self.dim is not None else t.mean()
        if self.kind == "sum":
            return t.sum(dim=self.dim) if self.dim is not None else t.sum()
        raise ValueError(self.kind)  # pragma: no cover


class GatherAction(Signal):
    """Select, for each row, the value at the given integer index along the last
    dim. This is ``q(obs)[actions]`` -- the action-value of the taken action."""

    def __init__(self, values: Signal, index: Signal) -> None:
        super().__init__(
            label=f"{values.label}[{index.label}]",
            parents=[values, index],
            provenance=combine_provenance(values.provenance, index.provenance),
            carries_grad=values.carries_grad,
        )

    def eval(self, ctx: Context) -> torch.Tensor:
        values = self.parents[0].eval(ctx)
        index = self.parents[1].eval(ctx).long()
        return values.gather(-1, index.unsqueeze(-1)).squeeze(-1)


class ConcatSignal(Signal):
    """Concatenate two signals along the last dimension."""

    def __init__(self, a: Signal, b: Signal) -> None:
        super().__init__(
            label=f"cat({a.label}, {b.label})",
            parents=[a, b],
            provenance=combine_provenance(a.provenance, b.provenance),
            carries_grad=a.carries_grad or b.carries_grad,
        )

    def eval(self, ctx: Context) -> torch.Tensor:
        return torch.cat([self.parents[0].eval(ctx), self.parents[1].eval(ctx)], dim=-1)


def concat(a: Any, b: Any) -> Signal:
    """Concatenate two signals along the last dimension."""
    return ConcatSignal(as_signal(a), as_signal(b))


class DataField(Signal):
    """A field (obs / action / reward / ...) of a materialized batch from a data
    source. Data never carries gradients and is tagged with the source's
    provenance."""

    def __init__(self, key: str, field: str, provenance: Provenance) -> None:
        super().__init__(
            label=f"{field}",
            parents=[],
            provenance=provenance,
            carries_grad=False,
        )
        self.key = key  # unique per-source key in the materialized context

    def eval(self, ctx: Context) -> torch.Tensor:
        return ctx[self.key]
