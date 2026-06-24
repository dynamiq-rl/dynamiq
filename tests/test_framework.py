"""Tests for framework-level features: ValueNetwork, LR schedulers, squeeze."""

import numpy as np
import pytest

import dynamiq as dq


def _buffer_filled(n=64):
    buf = dq.ReplayBuffer(capacity=1000, obs_dim=4)
    for _ in range(n):
        buf.add(np.random.randn(4), np.random.randint(2), 1.0, np.random.randn(4), False)
    return buf


# --- ValueNetwork -----------------------------------------------------------

def test_value_network_output_shape():
    import torch
    vf = dq.ValueNetwork("vf", obs_dim=4)
    mod = vf.instantiate()
    out = mod(torch.randn(8, 4))
    assert out.shape == (8, 1)


def test_value_network_in_ppo_graph():
    roll = dq.RolloutBuffer(capacity=256, obs_dim=4)
    for _ in range(256):
        roll.add(np.zeros(4, np.float32), 0, 1.0, False, 0.0, -0.69)
    roll.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)

    pi = dq.PolicyNetwork("pi", 4, 2)
    vf = dq.ValueNetwork("vf", 4)
    b = dq.RolloutSample(roll, n=64)

    dist = dq.Categorical(pi(b.obs))
    new_logp = dist.log_prob(b.action)
    pl = dq.loss.ppo_clip(new_logp, b.old_log_prob, b.advantage, clip=0.2)
    vl = dq.loss.mse(vf(b.obs).squeeze(-1), b.return_, weight=0.5)
    eb = dq.loss.maximize(dist.entropy(), weight=0.01)

    algo = dq.compile([pl, vl, eb], opt=dq.Adam(3e-4), device="cpu")
    m = algo.step()
    assert "loss" in m and algo.update_step == 1


# --- squeeze signal op ------------------------------------------------------

def test_squeeze_signal():
    import torch
    vf = dq.ValueNetwork("vf_sq", obs_dim=4)
    buf = _buffer_filled()
    b = dq.ReplaySample(buf, n=8)
    squeezed = vf(b.obs).squeeze(-1)
    assert "squeeze" in squeezed.label
    assert squeezed.carries_grad is True


# --- LR schedulers ----------------------------------------------------------

def test_cosine_annealing_decays_lr():
    buf = _buffer_filled()
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)

    algo = dq.compile(
        loss, opt=dq.Adam(1e-3), device="cpu",
        scheduler=dq.CosineAnnealing(T_max=100, eta_min=1e-5),
    )
    lr_before = algo.lr
    for _ in range(50):
        algo.step()
    lr_after = algo.lr
    assert lr_after < lr_before
    assert "lr" in algo.step()


def test_linear_decay_reaches_end():
    buf = _buffer_filled()
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)

    algo = dq.compile(
        loss, opt=dq.Adam(1e-3), device="cpu",
        scheduler=dq.LinearDecay(total_steps=100, end_factor=0.01),
    )
    lr_before = algo.lr
    for _ in range(100):
        algo.step()
    lr_after = algo.lr
    assert lr_after < lr_before
    assert lr_after == pytest.approx(1e-3 * 0.01, rel=0.1)


def test_step_decay():
    buf = _buffer_filled()
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)

    algo = dq.compile(
        loss, opt=dq.Adam(1e-3), device="cpu",
        scheduler=dq.StepDecay(step_size=10, gamma=0.5),
    )
    lr_before = algo.lr
    for _ in range(10):
        algo.step()
    assert algo.lr == pytest.approx(lr_before * 0.5, rel=1e-6)


def test_no_scheduler_still_works():
    buf = _buffer_filled()
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)
    algo = dq.compile(loss, opt=dq.Adam(1e-3), device="cpu")
    algo.step()
    assert algo.lr == 1e-3
