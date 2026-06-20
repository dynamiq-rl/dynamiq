"""Slice 1 tests: PPO and SAC express and type-check in the same primitives."""

import math

import numpy as np
import pytest

import dynamiq as dq
from dynamiq import DynamiqTypeError


def _fill_rollout(roll, n):
    for _ in range(n):
        roll.add(np.zeros(4, np.float32), 0, 1.0, False, 0.0, -0.69)
    roll.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)


def _fill_replay(buf, n):
    for _ in range(n):
        buf.add(np.random.randn(4), np.random.randint(2), 1.0, np.random.randn(4), False)


# --- PPO ------------------------------------------------------------------
def test_ppo_compiles_and_steps():
    roll = dq.RolloutBuffer(capacity=256, obs_dim=4)
    _fill_rollout(roll, 256)
    pi = dq.PolicyNetwork("pi", 4, 2)
    vf = dq.QNetwork("vf", 4, 1)
    b = dq.RolloutSample(roll, n=64)

    dist = dq.Categorical(pi(b.obs))
    new_logp = dist.log_prob(b.action)
    pl = dq.loss.ppo_clip(new_logp, b.old_log_prob, b.advantage, clip=0.2)
    vl = dq.loss.mse(vf(b.obs).max(-1), b.return_, weight=0.5)
    eb = dq.loss.maximize(dist.entropy(), weight=0.01)

    algo = dq.compile([pl, vl, eb], opt=dq.Adam(3e-4), device="cpu")
    assert dq.verify([pl, vl, eb]) == []
    m = algo.step()
    assert "loss" in m and algo.update_step == 1


def test_ppo_rejects_offpolicy_data():
    """The hallmark on-policy bug: PPO on replay data must be a compile error."""
    buf = dq.ReplayBuffer(1000, 4)
    pi = dq.PolicyNetwork("pi", 4, 2)
    b = dq.ReplaySample(buf, n=32)  # OFF_POLICY
    dist = dq.Categorical(pi(b.obs))
    new_logp = dist.log_prob(b.action)
    pl = dq.loss.ppo_clip(new_logp, b.reward, b.reward)  # off-policy everywhere
    with pytest.raises(DynamiqTypeError, match="on_policy"):
        dq.compile(pl, opt=dq.Adam(3e-4), device="cpu")


# --- SAC ------------------------------------------------------------------
def test_sac_compiles_steps_and_learns_temperature():
    buf = dq.ReplayBuffer(2000, 4)
    _fill_replay(buf, 300)

    q1 = dq.QNetwork("q1", 4, 2)
    q2 = dq.QNetwork("q2", 4, 2)
    q1t = dq.Target(q1, sync="soft", tau=0.01)
    q2t = dq.Target(q2, sync="soft", tau=0.01)
    pi = dq.PolicyNetwork("pi", 4, 2)
    log_alpha = dq.Parameter("log_alpha", 0.0)
    alpha = log_alpha.exp()
    b = dq.ReplaySample(buf, n=64)
    te = 0.5 * math.log(2)

    pin = dq.Categorical(pi(b.next_obs))
    pn, lpn = pin.probs(), pin.log_probs_all()
    minq_n = dq.minimum(q1t(b.next_obs), q2t(b.next_obs))
    v_next = (pn * (minq_n - alpha.detach() * lpn)).sum(-1)
    target = (b.reward + 0.99 * (1 - b.done) * v_next).detach()
    crit1 = dq.loss.mse(q1(b.obs)[b.action], target)
    crit2 = dq.loss.mse(q2(b.obs)[b.action], target)

    pic = dq.Categorical(pi(b.obs))
    p, lp = pic.probs(), pic.log_probs_all()
    minq = dq.minimum(q1(b.obs), q2(b.obs)).detach()
    actor = dq.loss.minimize(
        (p * (alpha.detach() * lp - minq)).sum(-1),
        requires_provenance=dq.Provenance.OFF_POLICY,
    )
    temp = dq.loss.minimize((p.detach() * (-log_alpha * (lp + te).detach())).sum(-1))

    algo = dq.compile([crit1, crit2, actor, temp], opt=dq.Adam(3e-4), device="cpu")
    assert algo.module("log_alpha") is not None  # temperature folded into optimizer
    before = float(algo.module("log_alpha").value.item())
    for _ in range(5):
        algo.step()
    after = float(algo.module("log_alpha").value.item())
    assert before != after  # the learnable temperature actually trained


def test_sac_target_without_detach_is_rejected():
    """If the SAC Bellman target isn't detached, the compiler must reject it."""
    buf = dq.ReplayBuffer(1000, 4)
    q = dq.QNetwork("q", 4, 2)
    qt = dq.Target(q, sync="soft", tau=0.01)
    pi = dq.PolicyNetwork("pi", 4, 2)
    b = dq.ReplaySample(buf, n=32)
    pin = dq.Categorical(pi(b.next_obs))
    # BUG: online policy probs in the target without detaching -> carries grad
    v_next = (pin.probs() * qt(b.next_obs)).sum(-1)
    target = b.reward + 0.99 * v_next  # not detached
    loss = dq.loss.mse(q(b.obs)[b.action], target)
    with pytest.raises(DynamiqTypeError, match="stop-gradient"):
        dq.compile(loss, opt=dq.Adam(3e-4), device="cpu")


# --- generic objective obligations ----------------------------------------
def test_minimize_detach_obligation_is_checked():
    buf = dq.ReplayBuffer(1000, 4)
    q = dq.QNetwork("q", 4, 2)
    b = dq.ReplaySample(buf, n=32)
    loss = dq.loss.minimize(q(b.obs).mean(), detach=(q(b.obs),))  # claim detached; it isn't
    with pytest.raises(DynamiqTypeError, match="stop-gradient"):
        dq.compile(loss, opt=dq.Adam(1e-3), device="cpu")
