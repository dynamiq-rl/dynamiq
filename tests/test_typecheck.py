"""The type checker is the product. These tests prove the classic RL footguns
become loud errors at compile time instead of silent training failures."""

import pytest

import dynamiq as dq
from dynamiq import DynamiqConfigError, DynamiqTypeError


def _buffer():
    return dq.ReplayBuffer(capacity=1000, obs_dim=4)


def test_valid_dqn_compiles():
    buf = _buffer()
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="hard", every=100)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)
    algo = dq.compile(loss, opt=dq.Adam(1e-3), device="cpu")
    assert algo.update_step == 0
    assert dq.verify(loss) == []


def test_missing_stop_gradient_is_rejected():
    """Bootstrapping from the ONLINE net (no Target / no detach) must fail."""
    buf = _buffer()
    q = dq.QNetwork("q", 4, 2)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * q(b.next_obs).max(-1)  # <- online net: carries grad
    loss = dq.loss.huber(q(b.obs)[b.action], target)
    with pytest.raises(DynamiqTypeError, match="carries gradients"):
        dq.compile(loss, opt=dq.Adam(1e-3), device="cpu")


def test_explicit_detach_fixes_missing_stop_gradient():
    buf = _buffer()
    q = dq.QNetwork("q", 4, 2)
    b = dq.ReplaySample(buf, n=32)
    target = (b.reward + 0.99 * q(b.next_obs).max(-1)).detach()
    loss = dq.loss.huber(q(b.obs)[b.action], target)
    dq.compile(loss, opt=dq.Adam(1e-3), device="cpu")  # should not raise


def test_offpolicy_data_in_onpolicy_loss_is_rejected():
    buf = _buffer()
    pi = dq.PolicyNetwork("pi", 4, 2)
    b = dq.ReplaySample(buf, n=32)  # OFF_POLICY
    logp = pi(b.obs).max(-1)
    adv = b.reward.detach()
    loss = dq.loss.policy_gradient(logp, adv)
    with pytest.raises(DynamiqTypeError, match="on-policy"):
        dq.compile(loss, opt=dq.Adam(1e-3), device="cpu")


def test_target_without_sync_rule_is_config_error():
    q = dq.QNetwork("q", 4, 2)
    with pytest.raises(DynamiqConfigError):
        dq.Target(q, sync="hard")  # missing `every`
    with pytest.raises(DynamiqConfigError):
        dq.Target(q, sync="soft")  # missing `tau`
    with pytest.raises(DynamiqConfigError):
        dq.Target(q, sync="nonsense")


def test_one_training_step_runs():
    buf = _buffer()
    # warm the buffer so sampling works
    import numpy as np

    for _ in range(64):
        buf.add(np.zeros(4), 0, 1.0, np.zeros(4), False)
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)
    algo = dq.compile(loss, opt=dq.Adam(1e-3), device="cpu")
    m = algo.step()
    assert "loss" in m and algo.update_step == 1


def test_gradient_clipping_by_norm():
    import numpy as np
    import torch

    buf = _buffer()
    for _ in range(64):
        buf.add(np.zeros(4), 0, 100.0, np.zeros(4), False)
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)
    algo = dq.compile(loss, opt=dq.Adam(1e-3), device="cpu", max_grad_norm=0.5)
    algo.step()
    total_norm = torch.nn.utils.clip_grad_norm_(algo._learnable, float("inf"))
    assert total_norm <= 0.5 + 1e-6


def test_gradient_clipping_by_value():
    import numpy as np
    import torch

    buf = _buffer()
    for _ in range(64):
        buf.add(np.zeros(4), 0, 100.0, np.zeros(4), False)
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    b = dq.ReplaySample(buf, n=32)
    target = b.reward + 0.99 * (1 - b.done) * qt(b.next_obs).max(-1)
    loss = dq.loss.huber(q(b.obs)[b.action], target)
    algo = dq.compile(loss, opt=dq.Adam(1e-3), device="cpu", max_grad_value=0.1)
    algo.step()
    # after step, grads are consumed; verify the clipping path doesn't error
    assert algo.update_step == 1
