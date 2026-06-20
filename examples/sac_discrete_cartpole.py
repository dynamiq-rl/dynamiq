"""Discrete SAC on CartPole-v1, written as a Dynamiq algorithm graph.

Same vocabulary as DQN and PPO, a third algorithm family. Note what the type
system is doing for us here:
  * the twin-Q minimum and the next-state policy go into a Bellman target that
    we `.detach()` -- the compiler *requires* it, so we can't leak gradients;
  * the actor objective uses `.detach()` on the Q-values so policy updates never
    touch the critics;
  * the temperature `log_alpha` is a learnable `dq.Parameter`, folded into the
    optimizer automatically.

Four objectives -- two critics, an actor, a temperature -- summed by the
compiler into one update.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import dynamiq as dq


def build_sac(obs_dim: int, n_actions: int, buffer: dq.ReplayBuffer, batch: int, gamma: float):
    q1 = dq.QNetwork("q1", obs_dim, n_actions, hidden=(128, 128))
    q2 = dq.QNetwork("q2", obs_dim, n_actions, hidden=(128, 128))
    q1t = dq.Target(q1, sync="soft", tau=0.01)
    q2t = dq.Target(q2, sync="soft", tau=0.01)
    pi = dq.PolicyNetwork("pi", obs_dim, n_actions, hidden=(128, 128))
    log_alpha = dq.Parameter("log_alpha", init=0.0)
    alpha = log_alpha.exp()

    b = dq.ReplaySample(buffer, n=batch)
    target_entropy = 0.5 * math.log(n_actions)

    # --- critic target: soft state-value at s' under the current policy ---
    pin = dq.Categorical(pi(b.next_obs))
    probs_n, logp_n = pin.probs(), pin.log_probs_all()
    minq_n = dq.minimum(q1t(b.next_obs), q2t(b.next_obs))
    v_next = (probs_n * (minq_n - alpha.detach() * logp_n)).sum(-1)
    target = (b.reward + gamma * (1.0 - b.done) * v_next).detach()

    crit1 = dq.loss.mse(q1(b.obs)[b.action], target)
    crit2 = dq.loss.mse(q2(b.obs)[b.action], target)

    # --- actor: minimize E_a[ pi(a|s) * (alpha*logpi - minQ) ] (Q detached) ---
    pic = dq.Categorical(pi(b.obs))
    probs, logp = pic.probs(), pic.log_probs_all()
    minq = dq.minimum(q1(b.obs), q2(b.obs)).detach()
    actor_term = (probs * (alpha.detach() * logp - minq)).sum(-1)
    actor = dq.loss.minimize(actor_term, requires_provenance=dq.Provenance.OFF_POLICY)

    # --- temperature: drive policy entropy toward target_entropy ---
    temp_term = (probs.detach() * (-log_alpha * (logp + target_entropy).detach())).sum(-1)
    temp = dq.loss.minimize(temp_term)

    algo = dq.compile([crit1, crit2, actor, temp], opt=dq.Adam(3e-4))
    return algo, pi


@torch.no_grad()
def act(pi, obs_t, greedy=False):
    logits = pi.module(obs_t)
    if greedy:
        return int(logits.argmax().item())
    return int(torch.distributions.Categorical(logits=logits).sample().item())


def main() -> None:
    import gymnasium as gym

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    gamma = 0.99
    buffer = dq.ReplayBuffer(capacity=50_000, obs_dim=obs_dim)
    algo, pi = build_sac(obs_dim, n_actions, buffer, batch=128, gamma=gamma)

    print(dq.explain(algo.losses))
    print(f"\ndevice: {algo.device}\n")

    total_steps, learn_start = 25_000, 1_000
    obs, _ = env.reset(seed=seed)
    ep_return, returns = 0.0, []

    for step in range(1, total_steps + 1):
        if step < learn_start:
            action = env.action_space.sample()
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=algo.device)
            action = act(pi, obs_t)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        if done:
            returns.append(ep_return)
            obs, _ = env.reset()
            ep_return = 0.0

        if step >= learn_start:
            algo.step()

        if step % 2_000 == 0 and returns:
            recent = np.mean(returns[-20:])
            print(f"step {step:6d} | mean return (last 20): {recent:6.1f}")

    final = np.mean(returns[-20:]) if returns else 0.0
    print(f"\nFinal mean return (last 20 episodes): {final:.1f}")
    if final >= 195.0:
        print("CartPole solved with discrete SAC -- third family, same primitives.")
    env.close()


if __name__ == "__main__":
    main()
