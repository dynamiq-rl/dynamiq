<p align="center">
  <img src="public/logo.png" alt="Dynamiq" width="420">
</p>

<h1 align="center">dynamiq</h1>

<p align="center">
  <b>Reinforcement-learning algorithms as a typed, compiled graph, not a hand-written training loop.</b>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#the-idea">The idea</a> ·
  <a href="#the-type-system">Type system</a> ·
  <a href="#worked-examples-with-the-math">The math</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/backend-PyTorch-ee4c2c" alt="pytorch">
  <img src="https://img.shields.io/badge/tests-11%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="status">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

---

## Why Dynamiq

Every mainstream deep-RL framework asks you to **write the training loop**: compute the
targets, compute the loss, call `backward()`, step the optimizer, sync the target
network, and *remember* to `stop_gradient` in exactly the right places. That is precisely
where RL bugs live, and they don't crash, they just quietly fail to learn:

- a bootstrap target that was never detached, silently back-propagating through the target;
- replay-buffer (off-policy) data fed into an on-policy objective like PPO;
- a target network that is never synchronized, or synchronized on the wrong schedule.

**Dynamiq flips the model.** You *declare* an algorithm as a graph of **typed signals**.
A compiler type-checks the graph, turning those silent failures into **compile-time
errors**, and then builds the optimized loop for you. And because an algorithm is now a
first-class **graph object** rather than opaque imperative code, it can be inspected,
linted, explained, and (on the roadmap) *searched*.

```python
import dynamiq as dq

q        = dq.QNetwork("q", obs_dim, n_actions)
q_target = dq.Target(q, sync="hard", every=500)        # a sync rule is mandatory
batch    = dq.ReplaySample(buffer, n=128)              # typed OFF_POLICY

target = batch.reward + gamma * (1 - batch.done) * q_target(batch.next_obs).max(-1)
loss   = dq.loss.huber(q(batch.obs)[batch.action], target)

algo = dq.compile(loss, opt=dq.Adam(2.5e-4))           # ← type-checks here
algo.step()                                            # one optimized update
```

That is a complete, correct DQN. `q_target(...)` is **auto-detached**, so the bootstrap
can never leak gradients. The target net **cannot be built without a sync rule**. The
replay sample is **typed off-policy**, so feeding it to an on-policy objective is a
compile error, not a 3 a.m. debugging session.

---

## Quickstart

```bash
git clone https://github.com/dynamiq-rl/dynamiq.git
cd dynamiq

python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate

pip install -e ".[examples,dev]"      # CPU wheel by default
# GPU: install the matching CUDA build, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Run the test suite and the examples:

```bash
pytest -q                               # 11 tests, all green
python examples/dqn_cartpole.py         # DQN: value-based, off-policy
python examples/ppo_cartpole.py         # PPO: policy-gradient, on-policy
python examples/sac_discrete_cartpole.py# SAC: actor-critic, off-policy
```

Dynamiq is **device-agnostic**: `dq.compile(..., device="cuda")` (or `"cpu"`); it
auto-detects CUDA when `device` is omitted.

---

## The idea

An algorithm is a directed graph of `Signal` nodes. You build it with ordinary Python
operators (`+`, `*`, indexing, `.max()`, `.detach()`, …); each node carries **two pieces
of static type information that ordinary tensors do not**:

| Static type | Meaning | What it prevents |
|---|---|---|
| `provenance` | `ON_POLICY` / `OFF_POLICY` / `AGNOSTIC` (the distribution the data came from) | off-policy data inside an on-policy objective; mixing distributions |
| `carries_grad` | whether evaluating the node is differentiably connected to learnable params | missing stop-gradient on a bootstrap/target |

These are computed **structurally as the graph is built**, so they cost nothing at runtime
and are available for static analysis. The pipeline is:

```
 declare graph ──▶ dq.compile ──┬─▶ type-check (provenance + grad obligations)
                                │
                                ├─▶ discover networks / targets / params / sources
                                │
                                └─▶ build optimizer + executable step()
```

### Building blocks

| Concept | What it is | API |
|---|---|---|
| **Signal** | a node in the algorithm graph | arithmetic, `.max`, `.clamp`, `.detach`, `s[a]` (gather) |
| **Network** | a learnable function approximator | `QNetwork`, `PolicyNetwork` |
| **Target** | a frozen, auto-detached, mandatorily-synced copy of a network | `Target(net, sync="hard", every=…)` / `sync="soft", tau=…` |
| **Source** | a *typed* data entry point | `ReplaySample` (off-policy), `RolloutSample` (on-policy + GAE) |
| **Distribution** | action distribution over a logits signal | `Categorical(logits)` → `log_prob`, `entropy`, `probs`, `log_probs_all` |
| **Parameter** | a single learnable scalar (e.g. SAC temperature) | `Parameter("log_alpha", init=0.0)` |
| **Loss** | an objective with typing *obligations*, plus a weight | `dq.loss.{huber, mse, ppo_clip, minimize, maximize, …}` |
| **compile** | type-check → runnable `Algorithm` | `dq.compile(objective, opt, device=None)` |

---

## The type system

The compiler enforces, **before any gradient step**:

| Classic RL bug | Dynamiq result |
|---|---|
| Bootstrap / target not detached | `DynamiqTypeError`: *"… must be a stop-gradient, but it still carries gradients."* |
| Off-policy data in an on-policy loss | `DynamiqTypeError`: *"requires on_policy data but … is off_policy."* |
| Target network with no sync rule | `DynamiqConfigError` at construction time |
| Mixing on-policy and off-policy data | `DynamiqTypeError`: *"… mixes on-policy and off-policy data."* |

Objectives carry their requirements as **obligations that reference specific graph
nodes**, so even hand-composed objectives (e.g. SAC's actor and temperature losses) are
fully checked. Non-fatal linting and introspection:

```python
dq.verify(objective)    # -> list[Finding]  (empty when clean)
print(dq.explain(objective))
# Dynamiq algorithm
#   objectives : 1
#     - huber(q(obs)[action], (reward + ...))
#   networks   : q
#   targets    : q_target
#   data       : off_policy
#   family     : value-based / off-policy (DQN/SAC-like)
```

---

## Worked examples (with the math)

Each algorithm below shows the standard equations and the **exact Dynamiq code** that
expresses them. The training/environment-interaction loop is orthogonal and lives in
[`examples/`](examples/).

### 1. DQN: value-based, off-policy

The Bellman target uses the **target network** $Q_{\theta^-}$ and is a stop-gradient; the
online network $Q_\theta$ is regressed toward it:

$$
y = r + \gamma\,(1-d)\,\max_{a'} Q_{\theta^-}(s', a')
\qquad
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\Big[\,\mathrm{Huber}\big(Q_\theta(s,a) - y\big)\Big]
$$

with periodic hard sync $\theta^- \leftarrow \theta$ every $C$ updates.

```python
q        = dq.QNetwork("q", obs_dim, n_actions)
q_target = dq.Target(q, sync="hard", every=500)
batch    = dq.ReplaySample(buffer, n=128)              # 𝒟 (off-policy)

target = batch.reward + gamma * (1 - batch.done) * q_target(batch.next_obs).max(-1)   # y
loss   = dq.loss.huber(q(batch.obs)[batch.action], target)                            # ℒ(θ)
algo   = dq.compile(loss, opt=dq.Adam(2.5e-4))
```

`q_target(...)` ⇒ $Q_{\theta^-}$ (auto-detached, so $y$ is a stop-gradient); `[batch.action]`
gathers $Q_\theta(s,a)$. Using the **online** net in `target` would make $y$ carry
gradients → compile error.

### 2. PPO: policy-gradient, on-policy

Advantages and returns come from **Generalized Advantage Estimation** over a rollout:

$$
\delta_t = r_t + \gamma\,(1-d_t)\,V(s_{t+1}) - V(s_t),
\qquad
\hat{A}_t = \sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l},
\qquad
R_t = \hat{A}_t + V(s_t)
$$

The objective is the **clipped surrogate** plus a value term and an entropy bonus, with
$\rho_t(\theta) = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)} = \exp\big(\log\pi_\theta - \log\pi_{\theta_\text{old}}\big)$:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = -\,\mathbb{E}_t\Big[\min\big(\rho_t\,\hat{A}_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big]
$$

$$
\mathcal{L}(\theta) = \mathcal{L}^{\text{CLIP}} + c_v\,\mathbb{E}_t\big[(V_\theta(s_t) - R_t)^2\big] - c_e\,\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot\mid s_t))\big]
$$

```python
batch    = dq.RolloutSample(rollout, n=256)            # ON_POLICY; advantage, return_, old_log_prob are data (detached)
dist     = dq.Categorical(pi(batch.obs))
new_logp = dist.log_prob(batch.action)                 # log πθ(a|s)

policy_loss   = dq.loss.ppo_clip(new_logp, batch.old_log_prob, batch.advantage, clip=0.2)   # ℒ^CLIP
value_loss    = dq.loss.mse(vf(batch.obs).max(-1), batch.return_, weight=0.5)               # c_v · MSE
entropy_bonus = dq.loss.maximize(dist.entropy(), weight=0.01)                               # − c_e · H

algo = dq.compile([policy_loss, value_loss, entropy_bonus], opt=dq.Adam(3e-4))
```

`ppo_clip` builds $\rho_t$, the `min`, and the clip *from primitives*, and requires
`ON_POLICY` data; passing a `ReplaySample` here is a compile error. GAE is computed by
`rollout.compute_gae(last_value, gamma, lam)` before the update epochs.

### 3. SAC (discrete): actor-critic, off-policy

Twin critics $Q_{\theta_1}, Q_{\theta_2}$, a stochastic policy $\pi_\phi$, and a learned
temperature $\alpha$. The **soft state value** at the next state and the critic target:

$$
V(s') = \sum_{a}\pi_\phi(a\mid s')\Big[\min_{i\in\{1,2\}} Q_{\bar\theta_i}(s',a) - \alpha\log\pi_\phi(a\mid s')\Big],
\qquad
y = r + \gamma\,(1-d)\,V(s')
$$

$$
\mathcal{L}_{Q_i} = \mathbb{E}\big[(Q_{\theta_i}(s,a) - y)^2\big]
\qquad i\in\{1,2\}
$$

The **actor** minimizes the expected soft value (critics detached), and the
**temperature** is driven toward a target entropy $\bar{\mathcal H}$:

$$
J_\pi(\phi) = \mathbb{E}_s\sum_a \pi_\phi(a\mid s)\big[\alpha\log\pi_\phi(a\mid s) - \min_i Q_{\theta_i}(s,a)\big],
\qquad
J_\alpha = \mathbb{E}_s\sum_a \pi_\phi(a\mid s)\big[-\alpha\big(\log\pi_\phi(a\mid s) + \bar{\mathcal H}\big)\big]
$$

with soft target updates $\bar\theta_i \leftarrow \tau\theta_i + (1-\tau)\bar\theta_i$.

```python
q1, q2   = dq.QNetwork("q1", o, a), dq.QNetwork("q2", o, a)
q1t, q2t = dq.Target(q1, "soft", tau=0.01), dq.Target(q2, "soft", tau=0.01)
pi        = dq.PolicyNetwork("pi", o, a)
log_alpha = dq.Parameter("log_alpha", 0.0);  alpha = log_alpha.exp()
b         = dq.ReplaySample(buffer, n=128)

# soft state value at s'  → Bellman target y  (fully detached, as the type system requires)
pin = dq.Categorical(pi(b.next_obs))
minq_n = dq.minimum(q1t(b.next_obs), q2t(b.next_obs))
v_next = (pin.probs() * (minq_n - alpha.detach() * pin.log_probs_all())).sum(-1)
target = (b.reward + gamma * (1 - b.done) * v_next).detach()                       # y
crit1  = dq.loss.mse(q1(b.obs)[b.action], target)                                  # ℒ_Q1
crit2  = dq.loss.mse(q2(b.obs)[b.action], target)                                  # ℒ_Q2

# actor (Q detached so policy never updates the critics)
pic   = dq.Categorical(pi(b.obs))
minq  = dq.minimum(q1(b.obs), q2(b.obs)).detach()
actor = dq.loss.minimize((pic.probs() * (alpha.detach()*pic.log_probs_all() - minq)).sum(-1),
                         requires_provenance=dq.Provenance.OFF_POLICY)              # J_π

# temperature
temp = dq.loss.minimize((pic.probs().detach() *
                         (-log_alpha * (pic.log_probs_all() + target_entropy).detach())).sum(-1))  # J_α

algo = dq.compile([crit1, crit2, actor, temp], opt=dq.Adam(3e-4))
```

`dq.minimum` is the twin-Q $\min$; `dq.Parameter` is the learned $\alpha$, folded into the
optimizer automatically. The compiler **requires** the Bellman target to be detached: the
`test_sac_target_without_detach_is_rejected` test exercises exactly the bug where it isn't.

---

## Results

Three different algorithm *families*, expressed in the same primitives, all solve
**CartPole-v1** (seed 0, CPU):

| Algorithm | Family | Mean return (last 20) | Example |
|---|---|---|---|
| DQN | value-based, off-policy | solves | [`dqn_cartpole.py`](examples/dqn_cartpole.py) |
| PPO | policy-gradient, on-policy | **≈ 447 / 500** | [`ppo_cartpole.py`](examples/ppo_cartpole.py) |
| SAC (discrete) | actor-critic, off-policy | **≈ 416 / 500** | [`sac_discrete_cartpole.py`](examples/sac_discrete_cartpole.py) |

---

## API reference

**Networks & parameters.** `QNetwork(name, obs_dim, n_actions, hidden=(128,128))`,
`PolicyNetwork(...)`, `Parameter(name, init=0.0)`.

**Targets.** `Target(source, sync="hard", every=N)` or `Target(source, sync="soft", tau=τ)`.
Calling a target always yields a detached signal.

**Sources.** `ReplayBuffer(capacity, obs_dim)` + `ReplaySample(buffer, n)` (fields:
`obs, action, reward, next_obs, done`, **OFF_POLICY**); `RolloutBuffer(capacity, obs_dim)`
(`.add`, `.compute_gae(last_value, gamma, lam)`, `.reset_storage`) + `RolloutSample(buffer, n)`
(fields: `obs, action, advantage, return_, old_log_prob`, **ON_POLICY**).

**Distributions.** `Categorical(logits)` provides `.log_prob(a)`, `.entropy()`, `.probs()`,
`.log_probs_all()`.

**Signal ops.** `+ - * /`, `-x`, `s[a]` (gather taken-action value), `.max(dim)`,
`.min`, `.mean`, `.sum`, `.log()`, `.exp()`, `.clamp(lo, hi)`, `.detach()`;
module-level `dq.minimum(a, b)`, `dq.maximum(a, b)`.

**Objectives** (all accept `weight=`). `dq.loss.huber(pred, target, delta=1.0)`,
`dq.loss.mse(pred, target)`, `dq.loss.policy_gradient(logp, adv)`,
`dq.loss.ppo_clip(new_logp, old_logp, adv, clip=0.2)`,
`dq.loss.minimize(scalar, detach=(...), requires_provenance=...)`, `dq.loss.maximize(scalar, ...)`.

**Compile & run.** `algo = dq.compile(objective_or_list, opt=dq.Adam(lr), device=None)`;
`algo.step() -> dict[str, float]`, `algo.module(name)`, `algo.update_step`, `algo.device`,
`algo.losses`.

**Static analysis.** `dq.verify(objective) -> list[Finding]`, `dq.explain(objective) -> str`.

---

## Project layout

```
dynamiq/
  signal.py     # the IR: Signal nodes, provenance + carries_grad typing, operators
  networks.py   # QNetwork / PolicyNetwork (learnable function approximators)
  targets.py    # Target: auto-detached, mandatorily-synced target networks
  sources.py    # ReplayBuffer/ReplaySample (off-policy), RolloutBuffer/RolloutSample + GAE (on-policy)
  dist.py       # Categorical action distribution as graph nodes
  params.py     # Parameter: learnable scalars (e.g. SAC temperature)
  losses.py     # objectives + node-referencing typing obligations
  compiler.py   # type-check -> instantiate -> executable Algorithm.step()
  verify.py     # verify() linting + explain() introspection
examples/       # dqn / ppo / sac, each learning CartPole end-to-end
tests/          # type-checker + learning-smoke tests (11, all green)
```

---

## Roadmap

- **Slice 2: algorithm search (the novel payoff).** Because an algorithm is a typed
  graph object, mutate/evolve objective graphs where every candidate is *valid by
  construction*: the type system rejects the nonsense a search would otherwise waste
  compute on. AutoRL over a typed IR.
- **Slice 3: production hardening.** Vectorized environments, continuous-action SAC
  (squashed-Gaussian policies), API stability, and benchmarks vs CleanRL / Stable-Baselines3.

---

## Testing

```bash
pytest -q
```

The suite is intentionally about the *product*: it proves the classic RL footguns become
loud compile-time errors (missing stop-grad, off-policy data in on-policy losses, unsynced
targets), and that all three families compile, step, and learn.

---

## Citation

```bibtex
@software{dynamiq,
  title  = {Dynamiq: Reinforcement-Learning Algorithms as a Typed, Compiled Graph},
  author = {Arunabh Bora},
  year   = {2026},
  url    = {https://github.com/dynamiq-rl/dynamiq}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
