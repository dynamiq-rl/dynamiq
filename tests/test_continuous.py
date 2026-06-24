"""Tests for continuous-action support: GaussianPolicy, SquashedNormal,
ContinuousQNetwork, ContinuousReplayBuffer, and continuous SAC compilation."""

import numpy as np
import pytest
import torch

import dynamiq as dq
from dynamiq import DynamiqTypeError


def _fill_continuous_buffer(buf, n):
    action_dim = buf.action.shape[1]
    for _ in range(n):
        buf.add(
            np.random.randn(buf.obs.shape[1]),
            np.random.randn(action_dim),
            np.random.randn(),
            np.random.randn(buf.obs.shape[1]),
            False,
        )


def test_gaussian_policy_output_shape():
    pi = dq.GaussianPolicy("pi", obs_dim=3, action_dim=2)
    mod = pi.instantiate()
    out = mod(torch.randn(4, 3))
    assert out.shape == (4, 4)  # 2 * action_dim


def test_continuous_q_output_shape():
    q = dq.ContinuousQNetwork("q", obs_dim=3, action_dim=2)
    mod = q.instantiate()
    out = mod(torch.randn(4, 5))  # obs_dim + action_dim = 5
    assert out.shape == (4, 1)


def test_squashed_normal_log_prob_is_finite():
    pi = dq.GaussianPolicy("pi", 3, 2)
    pi.instantiate()
    b = dq.ContinuousReplayBuffer(100, 3, 2)
    _fill_continuous_buffer(b, 50)
    sample = dq.ContinuousReplaySample(b, n=16)

    params_sig = pi(sample.obs)
    dist = dq.SquashedNormal(params_sig, action_dim=2)
    logp = dist.log_prob(sample.action)

    ctx = sample.materialize(torch.device("cpu"))
    result = logp.eval(ctx)
    assert result.shape == (16,)
    assert torch.isfinite(result).all()


def test_squashed_normal_rsample_in_range():
    pi = dq.GaussianPolicy("pi", 3, 2)
    pi.instantiate()
    b = dq.ContinuousReplayBuffer(100, 3, 2)
    _fill_continuous_buffer(b, 50)
    sample = dq.ContinuousReplaySample(b, n=16)

    params_sig = pi(sample.obs)
    dist = dq.SquashedNormal(params_sig, action_dim=2)
    action = dist.rsample()

    ctx = sample.materialize(torch.device("cpu"))
    result = action.eval(ctx)
    assert result.shape == (16, 2)
    assert (result.abs() < 1.0).all()


def test_concat_signal():
    b = dq.ContinuousReplayBuffer(100, 3, 2)
    _fill_continuous_buffer(b, 50)
    sample = dq.ContinuousReplaySample(b, n=8)

    cat = dq.concat(sample.obs, sample.action)
    ctx = sample.materialize(torch.device("cpu"))
    result = cat.eval(ctx)
    assert result.shape == (8, 5)  # 3 + 2


def test_continuous_sac_compiles_and_steps():
    obs_dim, action_dim = 3, 1
    buf = dq.ContinuousReplayBuffer(2000, obs_dim, action_dim)
    _fill_continuous_buffer(buf, 300)

    q1 = dq.ContinuousQNetwork("q1", obs_dim, action_dim)
    q2 = dq.ContinuousQNetwork("q2", obs_dim, action_dim)
    q1t = dq.Target(q1, sync="soft", tau=0.005)
    q2t = dq.Target(q2, sync="soft", tau=0.005)
    pi = dq.GaussianPolicy("pi", obs_dim, action_dim)
    log_alpha = dq.Parameter("log_alpha", 0.0)
    alpha = log_alpha.exp()
    b = dq.ContinuousReplaySample(buf, n=64)
    gamma = 0.99
    target_entropy = -float(action_dim)

    # critic target
    next_dist = dq.SquashedNormal(pi(b.next_obs), action_dim)
    next_a = next_dist.rsample()
    next_logp = next_dist.log_prob(next_a)
    nq1 = q1t(dq.concat(b.next_obs, next_a)).sum(-1)
    nq2 = q2t(dq.concat(b.next_obs, next_a)).sum(-1)
    min_nq = dq.minimum(nq1, nq2)
    target = (b.reward + gamma * (1.0 - b.done) * (min_nq - alpha.detach() * next_logp)).detach()
    crit1 = dq.loss.mse(q1(dq.concat(b.obs, b.action)).sum(-1), target)
    crit2 = dq.loss.mse(q2(dq.concat(b.obs, b.action)).sum(-1), target)

    # actor
    curr_dist = dq.SquashedNormal(pi(b.obs), action_dim)
    curr_a = curr_dist.rsample()
    curr_logp = curr_dist.log_prob(curr_a)
    q1_pi = q1(dq.concat(b.obs, curr_a)).sum(-1)
    q2_pi = q2(dq.concat(b.obs, curr_a)).sum(-1)
    min_q = dq.minimum(q1_pi, q2_pi).detach()
    actor = dq.loss.minimize(
        alpha.detach() * curr_logp - min_q,
        requires_provenance=dq.Provenance.OFF_POLICY,
    )

    # temperature
    temp = dq.loss.minimize(-log_alpha * (curr_logp + target_entropy).detach())

    algo = dq.compile([crit1, crit2, actor, temp], opt=dq.Adam(3e-4), device="cpu")
    m = algo.step()
    assert "loss" in m and algo.update_step == 1

    # temperature actually trains
    before = float(algo.module("log_alpha").value.item())
    for _ in range(5):
        algo.step()
    after = float(algo.module("log_alpha").value.item())
    assert before != after


def test_continuous_sac_online_q_in_target_is_rejected():
    """Using the ONLINE Q (not target Q) in the Bellman target must fail."""
    obs_dim, action_dim = 3, 1
    buf = dq.ContinuousReplayBuffer(100, obs_dim, action_dim)
    q = dq.ContinuousQNetwork("q", obs_dim, action_dim)
    b = dq.ContinuousReplaySample(buf, n=32)

    # BUG: using online q instead of target q -- carries grad
    target = b.reward + 0.99 * q(dq.concat(b.next_obs, b.action)).sum(-1)
    loss = dq.loss.mse(q(dq.concat(b.obs, b.action)).sum(-1), target)
    with pytest.raises(DynamiqTypeError, match="stop-gradient"):
        dq.compile(loss, opt=dq.Adam(3e-4), device="cpu")
