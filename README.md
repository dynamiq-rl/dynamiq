# Dynamiq

**RL algorithms as a typed, compiled graph — not as a hand-written training loop.**

Every other deep-RL framework asks you to *write the loop*: compute targets,
compute the loss, `backward()`, `step()`, sync the target net, remember to
`stop_gradient`. That is exactly where the bugs live — a missing stop-grad on a
bootstrap target, off-policy data fed into an on-policy objective, a target
network that silently never syncs. These don't crash; they just quietly fail to
learn.

Dynamiq flips it. You **declare** an algorithm as a graph of typed signals, and a
compiler type-checks it and builds the loop for you. Whole classes of RL bugs
become **compile-time errors** instead of 3-AM debugging sessions. And because an
algorithm is now *data* (a graph object), it can be inspected, linted, explained
— and, ahead, *searched*.

## The idea in four lines

```python
import dynamiq as dq

q        = dq.QNetwork("q", obs_dim, n_actions)
q_target = dq.Target(q, sync="hard", every=500)      # sync rule is mandatory
batch    = dq.ReplaySample(buffer, n=128)            # typed OFF_POLICY

target = batch.reward + gamma * (1 - batch.done) * q_target(batch.next_obs).max(-1)
loss   = dq.loss.huber(q(batch.obs)[batch.action], target)

algo = dq.compile(loss, opt=dq.Adam(2.5e-4))         # <-- type-checks here
algo.step()                                          # one optimized update
```

That is a complete, correct DQN. `q_target(...)` is **auto-detached**, so the
bootstrap can't leak gradients. The target net **can't be built without a sync
rule**. The replay sample is **typed off-policy**, so feeding it to an on-policy
objective is a compile error.

## What the type system catches (before training)

| Classic bug | In Dynamiq |
|---|---|
| Bootstrap target not detached | `DynamiqTypeError`: "target … still carries gradients" |
| Off-policy data in an on-policy loss | `DynamiqTypeError`: "requires ON_POLICY data, received OFF_POLICY" |
| Target network with no sync rule | `DynamiqConfigError` at construction |
| Mixing on- and off-policy data | `DynamiqTypeError`: "mixes on-policy and off-policy data" |

```python
dq.verify(loss)    # -> [] when clean, else a list of Findings (non-fatal lint)
dq.explain(loss)   # -> human-readable description + inferred algorithm family
```

## Status

Slices 0 and 1 are **done and validated**. Three different algorithm *families*,
expressed in the same primitives, all solve CartPole-v1:

| Algorithm | Family | Example | CartPole |
|---|---|---|---|
| DQN | value-based, off-policy | `examples/dqn_cartpole.py` | learns |
| PPO | policy-gradient, on-policy | `examples/ppo_cartpole.py` | ~447 / 500 |
| SAC (discrete) | actor-critic, off-policy | `examples/sac_discrete_cartpole.py` | ~416 / 500 |

What the framework provides:

- IR + provenance/grad type system (`dynamiq/signal.py`)
- networks, first-class target nodes, learnable scalar `Parameter`, typed
  on/off-policy sources with GAE (`networks.py`, `targets.py`, `params.py`, `sources.py`)
- action distributions (`dist.py`)
- composable objectives with node-referencing typing obligations, incl. a generic
  `minimize` (`losses.py`)
- compiler: static type-check → weighted-sum executable loop (`compiler.py`)
- `verify` / `explain` static analysis (`verify.py`)
- type-checker suite proving footguns are caught across all three families
  (`tests/` — 11 tests, all green)

Roadmap: **Slice 2** `dq.search` over the algorithm graph (the AutoRL payoff —
mutate/evolve objective graphs, type-checked by construction) · **Slice 3**
vectorized envs, continuous-action SAC, API stability, benchmarks vs CleanRL/SB3.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[examples,dev]"   # add CUDA torch wheel for GPU
.venv/Scripts/python -m pytest -q
.venv/Scripts/python examples/dqn_cartpole.py
```
