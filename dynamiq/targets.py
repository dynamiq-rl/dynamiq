"""Target networks as first-class, *mandatorily-synced*, auto-detached nodes.

In hand-written RL code a target network is just another module you have to
remember to (a) detach when bootstrapping and (b) sync on some schedule. Forget
either and training silently breaks.

In Dynamiq a ``Target`` is a node:
  * calling it ALWAYS produces a detached signal (``carries_grad=False``), so a
    bootstrap built from it can never accidentally leak gradients into the
    target;
  * it cannot be constructed without a sync rule -- that's a config error, not a
    runtime surprise.
"""

from __future__ import annotations

import copy

import torch

from .exceptions import DynamiqConfigError
from .networks import Network
from .signal import Context, Signal


class TargetApply(Signal):
    def __init__(self, target: "Target", x: Signal) -> None:
        super().__init__(
            label=f"{target.name}({x.label})",
            parents=[x],
            provenance=x.provenance,
            carries_grad=False,  # the whole reason Target exists
        )
        self.target = target

    def eval(self, ctx: Context) -> torch.Tensor:
        with torch.no_grad():
            return self.target.module(self.parents[0].eval(ctx))


class Target:
    """A frozen, periodically-synced copy of a source :class:`Network`.

    Parameters
    ----------
    source:
        The online network this target tracks.
    sync:
        ``"hard"`` (copy weights every ``every`` updates) or ``"soft"``
        (Polyak average with coefficient ``tau`` every update).
    every:
        Required for hard sync. Number of updates between full copies.
    tau:
        Required for soft sync. Polyak coefficient in (0, 1].
    """

    def __init__(
        self,
        source: Network,
        sync: str,
        every: int | None = None,
        tau: float | None = None,
    ) -> None:
        if sync == "hard":
            if not (isinstance(every, int) and every > 0):
                raise DynamiqConfigError(
                    "Target(sync='hard') requires a positive integer `every` "
                    "(updates between weight copies)."
                )
        elif sync == "soft":
            if not (isinstance(tau, (int, float)) and 0.0 < tau <= 1.0):
                raise DynamiqConfigError(
                    "Target(sync='soft') requires `tau` in (0, 1] (Polyak coefficient)."
                )
        else:
            raise DynamiqConfigError(
                f"Unknown sync rule {sync!r}; expected 'hard' or 'soft'."
            )

        self.source = source
        self.name = f"{source.name}_target"
        self.sync = sync
        self.every = every
        self.tau = tau
        self.module: torch.nn.Module | None = None

    def instantiate(self) -> torch.nn.Module:
        """Create the target module as a frozen clone of the source."""
        src = self.source.instantiate()
        self.module = copy.deepcopy(src)
        for p in self.module.parameters():
            p.requires_grad_(False)
        return self.module

    def maybe_sync(self, update_step: int) -> None:
        """Apply the sync rule. Called by the compiled algorithm after each update."""
        assert self.module is not None and self.source.module is not None
        if self.sync == "hard":
            if update_step % self.every == 0:
                self.module.load_state_dict(self.source.module.state_dict())
        else:  # soft
            with torch.no_grad():
                for tp, sp in zip(
                    self.module.parameters(), self.source.module.parameters()
                ):
                    tp.mul_(1.0 - self.tau).add_(self.tau * sp)

    def __call__(self, x: Signal) -> Signal:
        return TargetApply(self, x)

    def __repr__(self) -> str:
        rule = f"hard/every={self.every}" if self.sync == "hard" else f"soft/tau={self.tau}"
        return f"<Target {self.name} [{rule}]>"
