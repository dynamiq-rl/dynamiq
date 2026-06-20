"""PPO on CartPole-v1, written as a Dynamiq algorithm graph.

The *algorithm* is `build_ppo`: a clipped-surrogate policy objective, a value
regression, and an entropy bonus -- three objectives summed by the compiler. The
on-policy nature is enforced by the type system: `ppo_clip` requires ON_POLICY
data, and the rollout source provides exactly that. Old log-probs and advantages
are rollout *data*, hence detached by construction.
"""

from __future__ import annotations

import numpy as np
import torch

import dynamiq as dq


def build_ppo(obs_dim: int, n_actions: int, rollout: dq.RolloutBuffer, minibatch: int):
    pi = dq.PolicyNetwork("pi", obs_dim, n_actions, hidden=(64, 64))
    vf = dq.QNetwork("vf", obs_dim, 1, hidden=(64, 64))  # scalar value head
    batch = dq.RolloutSample(rollout, n=minibatch)

    dist = dq.Categorical(pi(batch.obs))
    new_logp = dist.log_prob(batch.action)

    policy_loss = dq.loss.ppo_clip(new_logp, batch.old_log_prob, batch.advantage, clip=0.2)
    value_loss = dq.loss.mse(vf(batch.obs).max(-1), batch.return_, weight=0.5)
    entropy_bonus = dq.loss.maximize(dist.entropy(), weight=0.01)

    algo = dq.compile([policy_loss, value_loss, entropy_bonus], opt=dq.Adam(3e-4))
    return algo, pi, vf


@torch.no_grad()
def act(pi, vf, obs_t):
    logits = pi.module(obs_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    value = vf.module(obs_t).squeeze(-1)
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def main() -> None:
    import gymnasium as gym

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    gamma, lam = 0.99, 0.95
    horizon, minibatch, epochs = 2048, 256, 8
    rollout = dq.RolloutBuffer(capacity=horizon, obs_dim=obs_dim)
    algo, pi, vf = build_ppo(obs_dim, n_actions, rollout, minibatch)

    print(dq.explain(algo.losses))
    print(f"\ndevice: {algo.device}\n")

    total_iters = 30
    obs, _ = env.reset(seed=seed)
    ep_return, returns = 0.0, []

    for it in range(1, total_iters + 1):
        rollout.reset_storage()
        for _ in range(horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=algo.device)
            action, logp, value = act(pi, vf, obs_t)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rollout.add(obs, action, reward, terminated, value, logp)
            obs = next_obs
            ep_return += reward
            if done:
                returns.append(ep_return)
                obs, _ = env.reset()
                ep_return = 0.0

        with torch.no_grad():
            last_v = float(vf.module(torch.as_tensor(obs, dtype=torch.float32, device=algo.device)).item())
        rollout.compute_gae(last_v, gamma, lam)

        updates = epochs * (horizon // minibatch)
        for _ in range(updates):
            algo.step()

        recent = np.mean(returns[-20:]) if returns else 0.0
        print(f"iter {it:3d} | episodes {len(returns):4d} | mean return (last 20): {recent:6.1f}")

    final = np.mean(returns[-20:]) if returns else 0.0
    print(f"\nFinal mean return (last 20 episodes): {final:.1f}")
    if final >= 195.0:
        print("CartPole solved with PPO -- same primitives, different family.")
    env.close()


if __name__ == "__main__":
    main()
