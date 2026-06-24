"""Data sources -- the *typed* entry points for data into an algorithm graph.

A source's provenance propagates through every signal derived from it, which is
what lets the compiler reject (say) replay-buffer data feeding an on-policy
objective. Off-policy data comes from a :class:`ReplayBuffer` via
:class:`ReplaySample`; on-policy data comes from a :class:`RolloutBuffer` (with
GAE) via :class:`RolloutSample`.
"""

from __future__ import annotations

import numpy as np
import torch

from .signal import Context, DataField, Provenance


class Source:
    """Base for data sources. Exposes named fields as typed Signals and knows
    how to materialize a concrete batch each update."""

    provenance: Provenance

    def __init__(self, key: str, fields: tuple[str, ...]) -> None:
        self._key = key
        self._fields = fields
        for field in fields:
            df = DataField(key=f"{key}/{field}", field=field, provenance=self.provenance)
            df.source = self  # lets the compiler discover sources from the graph
            setattr(self, field, df)

    def materialize(self, device: torch.device) -> Context:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# Off-policy: replay buffer
# --------------------------------------------------------------------------
class ReplayBuffer:
    """A simple ring buffer of transitions for off-policy learning."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self._idx = 0
        self._full = False

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self._idx
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._full = self._full or self._idx == 0

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def sample(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self), size=n)
        t = lambda a: torch.as_tensor(a[idx], device=device)
        return {
            "obs": t(self.obs),
            "action": t(self.action),
            "reward": t(self.reward),
            "next_obs": t(self.next_obs),
            "done": t(self.done),
        }


class ContinuousReplayBuffer:
    """A ring buffer for continuous-action transitions (float actions)."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self._idx = 0
        self._full = False

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self._idx
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._full = self._full or self._idx == 0

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def sample(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self), size=n)
        t = lambda a: torch.as_tensor(a[idx], device=device)
        return {
            "obs": t(self.obs),
            "action": t(self.action),
            "reward": t(self.reward),
            "next_obs": t(self.next_obs),
            "done": t(self.done),
        }


class ContinuousReplaySample(Source):
    """Sample ``n`` transitions from a continuous replay buffer. Provenance: OFF_POLICY."""

    provenance = Provenance.OFF_POLICY
    _counter = 0

    def __init__(self, buffer: ContinuousReplayBuffer, n: int) -> None:
        ContinuousReplaySample._counter += 1
        super().__init__(
            key=f"cont_replay{ContinuousReplaySample._counter}",
            fields=("obs", "action", "reward", "next_obs", "done"),
        )
        self.buffer = buffer
        self.n = n

    def materialize(self, device: torch.device) -> Context:
        batch = self.buffer.sample(self.n, device)
        return {f"{self._key}/{k}": v for k, v in batch.items()}


class ReplaySample(Source):
    """Sample ``n`` transitions from a replay buffer. Provenance: OFF_POLICY."""

    provenance = Provenance.OFF_POLICY
    _counter = 0

    def __init__(self, buffer: ReplayBuffer, n: int) -> None:
        ReplaySample._counter += 1
        super().__init__(
            key=f"replay{ReplaySample._counter}",
            fields=("obs", "action", "reward", "next_obs", "done"),
        )
        self.buffer = buffer
        self.n = n

    def materialize(self, device: torch.device) -> Context:
        batch = self.buffer.sample(self.n, device)
        return {f"{self._key}/{k}": v for k, v in batch.items()}


# --------------------------------------------------------------------------
# On-policy: rollout buffer with GAE
# --------------------------------------------------------------------------
class RolloutBuffer:
    """Fixed-horizon storage of on-policy transitions, with GAE advantage /
    return computation. Filled fresh each policy iteration, then cleared."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.reset_storage()

    def reset_storage(self) -> None:
        c, d = self.capacity, self.obs_dim
        self.obs = np.zeros((c, d), dtype=np.float32)
        self.action = np.zeros((c,), dtype=np.int64)
        self.reward = np.zeros((c,), dtype=np.float32)
        self.done = np.zeros((c,), dtype=np.float32)
        self.value = np.zeros((c,), dtype=np.float32)
        self.log_prob = np.zeros((c,), dtype=np.float32)
        self.advantage = np.zeros((c,), dtype=np.float32)
        self.return_ = np.zeros((c,), dtype=np.float32)
        self._n = 0

    def add(self, obs, action, reward, done, value, log_prob) -> None:
        i = self._n
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = float(done)
        self.value[i] = value
        self.log_prob[i] = log_prob
        self._n += 1

    def full(self) -> bool:
        return self._n >= self.capacity

    def __len__(self) -> int:
        return self._n

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> None:
        """Generalized Advantage Estimation over the stored rollout."""
        n = self._n
        adv = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.value[t + 1]
            next_nonterminal = 1.0 - self.done[t]
            delta = self.reward[t] + gamma * next_value * next_nonterminal - self.value[t]
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantage[t] = adv
            self.return_[t] = adv + self.value[t]

    def sample(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._n, size=n)
        adv = self.advantage[idx]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # per-minibatch normalization
        t = lambda a: torch.as_tensor(a, device=device)
        return {
            "obs": t(self.obs[idx]),
            "action": t(self.action[idx]),
            "advantage": t(adv.astype(np.float32)),
            "return_": t(self.return_[idx]),
            "old_log_prob": t(self.log_prob[idx]),
        }


class RolloutSample(Source):
    """Sample a minibatch from an on-policy rollout. Provenance: ON_POLICY."""

    provenance = Provenance.ON_POLICY
    _counter = 0

    def __init__(self, buffer: RolloutBuffer, n: int) -> None:
        RolloutSample._counter += 1
        super().__init__(
            key=f"rollout{RolloutSample._counter}",
            fields=("obs", "action", "advantage", "return_", "old_log_prob"),
        )
        self.buffer = buffer
        self.n = n

    def materialize(self, device: torch.device) -> Context:
        batch = self.buffer.sample(self.n, device)
        return {f"{self._key}/{k}": v for k, v in batch.items()}
