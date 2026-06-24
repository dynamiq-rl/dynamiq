"""Continuous SAC on Pendulum-v1, written as a Dynamiq algorithm graph.

This demonstrates continuous-action support: a squashed Gaussian policy
(tanh-Normal), continuous Q-networks Q(s,a), and a continuous replay buffer.
The type system enforces the same guarantees as discrete SAC -- detached
Bellman targets, off-policy provenance, auto-synced target networks.
"""

from __future__ import annotations

import numpy as np
import torch

import dynamiq as dq


def build_sac(obs_dim: int, action_dim: int, buffer: dq.ContinuousReplayBuffer,
              batch: int, gamma: float, action_scale: float):
    q1 = dq.ContinuousQNetwork("q1", obs_dim, action_dim, hidden=(256, 256))
    q2 = dq.ContinuousQNetwork("q2", obs_dim, action_dim, hidden=(256, 256))
    q1t = dq.Target(q1, sync="soft", tau=0.005)
    q2t = dq.Target(q2, sync="soft", tau=0.005)
    pi = dq.GaussianPolicy("pi", obs_dim, action_dim, hidden=(256, 256))
    log_alpha = dq.Parameter("log_alpha", init=0.0)
    alpha = log_alpha.exp()

    b = dq.ContinuousReplaySample(buffer, n=batch)
    target_entropy = -float(action_dim)

    # --- critic target ---
    # sample next action from current policy at s'
    next_dist = dq.SquashedNormal(pi(b.next_obs), action_dim)
    next_action = next_dist.rsample()
    next_logp = next_dist.log_prob(next_action)
    next_q1 = q1t(dq.concat(b.next_obs, next_action)).sum(-1)
    next_q2 = q2t(dq.concat(b.next_obs, next_action)).sum(-1)
    min_next_q = dq.minimum(next_q1, next_q2)
    target = (b.reward + gamma * (1.0 - b.done) * (min_next_q - alpha.detach() * next_logp)).detach()

    crit1 = dq.loss.mse(q1(dq.concat(b.obs, b.action)).sum(-1), target)
    crit2 = dq.loss.mse(q2(dq.concat(b.obs, b.action)).sum(-1), target)

    # --- actor ---
    curr_dist = dq.SquashedNormal(pi(b.obs), action_dim)
    curr_action = curr_dist.rsample()
    curr_logp = curr_dist.log_prob(curr_action)
    q1_pi = q1(dq.concat(b.obs, curr_action)).sum(-1)
    q2_pi = q2(dq.concat(b.obs, curr_action)).sum(-1)
    min_q_pi = dq.minimum(q1_pi, q2_pi).detach()
    actor = dq.loss.minimize(
        (alpha.detach() * curr_logp - min_q_pi),
        requires_provenance=dq.Provenance.OFF_POLICY,
    )

    # --- temperature ---
    temp = dq.loss.minimize(
        (-log_alpha * (curr_logp + target_entropy).detach()),
    )

    algo = dq.compile([crit1, crit2, actor, temp], opt=dq.Adam(3e-4))
    return algo, pi, action_scale


@torch.no_grad()
def act(pi_module, obs_t, action_scale, deterministic=False):
    params = pi_module(obs_t)
    mean, log_std = params.chunk(2, dim=-1)
    if deterministic:
        return torch.tanh(mean) * action_scale
    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    action = torch.tanh(dist.sample())
    return (action * action_scale).squeeze(0)


def main() -> None:
    import gymnasium as gym

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    gamma = 0.99
    buffer = dq.ContinuousReplayBuffer(capacity=100_000, obs_dim=obs_dim, action_dim=action_dim)
    algo, pi, a_scale = build_sac(obs_dim, action_dim, buffer, batch=256, gamma=gamma,
                                   action_scale=action_scale)

    print(dq.explain(algo.losses))
    print(f"\ndevice: {algo.device}\n")

    total_steps, learn_start = 30_000, 1_000
    obs, _ = env.reset(seed=seed)
    ep_return, returns = 0.0, []

    for step in range(1, total_steps + 1):
        if step < learn_start:
            action = env.action_space.sample()
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=algo.device).unsqueeze(0)
            action = act(pi, obs_t, a_scale).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # store normalized action (in [-1, 1]) for the policy
        buffer.add(obs, action / a_scale, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        if done:
            returns.append(ep_return)
            obs, _ = env.reset()
            ep_return = 0.0

        if step >= learn_start:
            algo.step()

        if step % 5_000 == 0 and returns:
            recent = np.mean(returns[-20:])
            print(f"step {step:6d} | mean return (last 20): {recent:7.1f}")

    final = np.mean(returns[-20:]) if returns else 0.0
    print(f"\nFinal mean return (last 20 episodes): {final:.1f}")
    if final >= -300.0:
        print("Pendulum improving with continuous SAC -- continuous actions work.")
    env.close()


if __name__ == "__main__":
    main()
