"""Static analysis over an algorithm graph -- the non-fatal counterpart to the
compiler's type checker, plus a human-readable explanation.

``verify`` returns findings instead of raising, so you can lint an algorithm.
``explain`` describes what an objective graph *is* -- because the algorithm is
data, we can introspect it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .losses import Detached, Loss, RequiresProv
from .networks import NetworkApply
from .signal import Provenance, Signal
from .targets import TargetApply


@dataclass
class Finding:
    severity: str  # "error" | "warning" | "info"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def _all_nodes(losses: list[Loss]) -> list[Signal]:
    seen: dict[int, Signal] = {}
    out: list[Signal] = []
    for loss in losses:
        for n in loss.walk():
            if id(n) not in seen:
                seen[id(n)] = n
                out.append(n)
    return out


def verify(objective: Loss | list[Loss]) -> list[Finding]:
    """Lint an objective graph. Returns a list of findings (possibly empty)."""
    losses = objective if isinstance(objective, list) else [objective]
    nodes = _all_nodes(losses)
    findings: list[Finding] = []

    for node in nodes:
        if node.provenance is Provenance.CONFLICT:
            findings.append(
                Finding("error", f"`{node.label}` mixes on-policy and off-policy data.")
            )

    for loss in losses:
        for ob in loss.obligations:
            if isinstance(ob, Detached) and ob.node.carries_grad:
                findings.append(
                    Finding("error", f"`{ob.node.label}` in `{loss.label}` is not detached.")
                )
            elif isinstance(ob, RequiresProv) and ob.node.provenance is not ob.required:
                findings.append(
                    Finding(
                        "error",
                        f"`{loss.label}` needs {ob.required} data, got {ob.node.provenance}.",
                    )
                )

    # Heuristic warning: a target that bootstraps from an *online* network
    # estimate (not pure data, not a Target) is a known source of instability.
    uses_target = any(isinstance(n, TargetApply) for n in nodes)
    bootstraps_online_net = any(
        any(isinstance(n, NetworkApply) for n in ob.node.walk())
        for l in losses
        for ob in l.obligations
        if isinstance(ob, Detached)
    )
    if bootstraps_online_net and not uses_target:
        findings.append(
            Finding(
                "warning",
                "objective bootstraps from an online network estimate with no "
                "Target network; value estimates may be unstable (consider dq.Target).",
            )
        )
    return findings


def explain(objective: Loss | list[Loss]) -> str:
    """Return a human-readable description of an objective graph."""
    losses = objective if isinstance(objective, list) else [objective]
    nodes = _all_nodes(losses)

    nets = sorted({n.net.name for n in nodes if isinstance(n, NetworkApply)})
    targets = sorted({n.target.name for n in nodes if isinstance(n, TargetApply)})
    provs = {n.provenance for n in nodes} - {Provenance.AGNOSTIC}

    lines = ["Dynamiq algorithm"]
    lines.append(f"  objectives : {len(losses)}")
    for l in losses:
        lines.append(f"    - {l.label}")
    lines.append(f"  networks   : {', '.join(nets) or '(none)'}")
    lines.append(f"  targets    : {', '.join(targets) or '(none)'}")
    lines.append(f"  data       : {', '.join(str(p) for p in provs) or 'agnostic'}")

    # Best-effort family inference.
    def _requires(p: Provenance) -> bool:
        return any(
            isinstance(o, RequiresProv) and o.required is p
            for l in losses
            for o in l.obligations
        )

    has_bootstrap = any(isinstance(o, Detached) for l in losses for o in l.obligations)
    family = "custom"
    if _requires(Provenance.ON_POLICY):
        family = "policy-gradient / on-policy (PPO/A2C-like)"
    elif targets and has_bootstrap and Provenance.OFF_POLICY in provs:
        family = "value-based / off-policy (DQN/SAC-like)"
    lines.append(f"  family     : {family}")
    return "\n".join(lines)
