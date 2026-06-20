"""DQN on CartPole-v1, written entirely as a Dynamiq algorithm graph.

Note what is NOT here: no hand-written loss/target/backward bookkeeping, no
manual stop-gradient, no manual target-sync logic. The *algorithm* is the four
lines in `build_dqn`; everything else is the standard environment-interaction
loop, which is orthogonal to the algorithm and the same for every value-based
method.
"""

from __future__ import annotations

import numpy as np
import torch

import dynamiq as dq


def build_dqn(obs_dim: int, n_actions: int, buffer: dq.ReplayBuffer, gamma: float):
    q = dq.QNetwork("q", obs_dim, n_actions, hidden=(128, 128))
    q_target = dq.Target(q, sync="hard", every=500)
    batch = dq.ReplaySample(buffer, n=128)

    # Bellman target. q_target(...) is auto-detached; (1 - done) masks terminals.
    target = batch.reward + gamma * (1.0 - batch.done) * q_target(batch.next_obs).max(-1)
    objective = dq.loss.huber(q(batch.obs)[batch.action], target)

    algo = dq.compile(objective, opt=dq.Adam(2.5e-4))
    return algo, q


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
    algo, q = build_dqn(obs_dim, n_actions, buffer, gamma)

    print(dq.explain(algo.losses))
    print(f"\ndevice: {algo.device}\n")

    total_steps = 25_000
    learn_start = 1_000
    eps_start, eps_end, eps_decay = 1.0, 0.05, 10_000

    obs, _ = env.reset(seed=seed)
    ep_return, returns = 0.0, []

    for step in range(1, total_steps + 1):
        eps = eps_end + (eps_start - eps_end) * max(0.0, (eps_decay - step) / eps_decay)
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=algo.device)
                action = int(q.module(obs_t).argmax().item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, terminated)  # store true terminal
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
            print(f"step {step:6d} | eps {eps:.2f} | mean return (last 20): {recent:6.1f}")

    final = np.mean(returns[-20:]) if returns else 0.0
    print(f"\nFinal mean return (last 20 episodes): {final:.1f}")
    if final >= 195.0:
        print("CartPole solved.  The paradigm works end-to-end.")
    env.close()


if __name__ == "__main__":
    main()
