"""E22: discrete SAC, sb3-contrib RecurrentPPO and stateful policies in the common runner."""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import optuna
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import tune_rl                                  # noqa: E402
from rltrade.agents import agent_policy, make_agent          # noqa: E402
from rltrade.env import EnvConfig, Market, run_policy        # noqa: E402
from rltrade.runner import RunSpec, backtest_saved, run      # noqa: E402
from tests.test_e16 import _prepared                         # noqa: E402

TINY_SAC = dict(net_dims=(16, 16), num_envs=4, learning_starts=200, batch_size=32, buffer_size=5_000)
TINY_RPPO = dict(net_dims=(16,), lstm_hidden=8, num_envs=4, rollout_len=32, minibatch=64, epochs=2)
HAS_SB3 = importlib.util.find_spec("sb3_contrib") is not None


def _spec(agent, hp, **kw):
    env = EnvConfig(state="A", episode_bars=300, start_skip=10)
    base = dict(experiment="t22", agent=agent, seed=0, env=env, agent_hp=hp, total_steps=2_000,
                eval_every=1_000, val_chunks=2, min_val_trades=1)
    base.update(kw)
    return RunSpec(**base)


def test_sac_trains_and_saved_run_reproduces():
    d, out = _prepared(), Path(tempfile.mkdtemp())
    row = run(_spec("sac", TINY_SAC, fallback="cash", fallback_min_score=-1e9), d, d, out, device="cpu",
              verbose=False, eval_splits=("val",))
    assert row["fallback_used"] is False and row["val_num_trades"] >= 0
    mets, _ = backtest_saved(out / "models" / "t22" / "sac_A_net_asset_change_s0", ["val"], device="cpu")
    assert np.isclose(mets["val"]["sharpe"], row["val_sharpe"]) and mets["val"]["num_trades"] == row["val_num_trades"]


def test_sac_update_moves_alpha_towards_the_entropy_target():
    ag = make_agent("sac", 10, 3, "cpu", 0, **{**TINY_SAC, "target_entropy_ratio": 0.2, "lr": 1e-2})
    obs = th.randn(500, 10)
    ag.replay.add(obs, th.randint(0, 3, (500,)), th.randn(500), th.randn(500, 10))
    a0 = float(ag.alpha)
    for _ in range(50):
        ag._train_step()
    # the initial policy is near uniform (entropy ~ log 3 > 0.2 log 3), so alpha must fall
    assert float(ag.alpha) < a0
    sd = ag.state_dict()
    ag2 = make_agent("sac", 10, 3, "cpu", 1, **TINY_SAC)
    ag2.load_state_dict(sd)
    assert th.equal(ag.act(obs), ag2.act(obs)) and float(ag2.alpha) == float(ag.alpha)


class _CountingPolicy:
    """Records how many steps it has seen since its last reset."""

    def __init__(self):
        self.resets, self.seen = 0, 0

    def reset(self):
        self.resets += 1
        self.seen = 0

    def __call__(self, obs):
        self.seen += 1
        return th.ones(obs.shape[0], dtype=th.long)


def test_run_policy_resets_a_stateful_policy_every_backtest():
    d = _prepared()
    cfg = EnvConfig(state="A")
    m = Market(d, d, "val", cfg, th.device("cpu"))
    pol = _CountingPolicy()
    a = run_policy(pol, m, cfg, chunks=1)
    first = pol.seen
    run_policy(pol, m, cfg, chunks=4)
    assert pol.resets == 2 and first == len(a["returns"]) and pol.seen < first


def test_search_spaces_build_valid_agents():
    for agent, params in [("sac", {"lr": 1e-4, "gamma": 0.99, "net": "small", "batch_size": 256, "replay_ratio": 0.25,
                                   "tau": 0.005, "target_entropy_ratio": 0.4, "reward_scale": 10.0}),
                          ("rppo", {"lr": 1e-4, "gamma": 0.99, "net": "small", "entropy_coef": 0.01, "clip": 0.2,
                                    "gae_lambda": 0.95, "rollout_len": 128, "minibatch": 1024, "epochs": 6,
                                    "lstm_hidden": 128, "reward_scale": 10.0})]:
        hp, env = tune_rl.suggest(optuna.trial.FixedTrial(params), agent)
        assert env == {"reward_scale": 10.0} and hp["net_dims"] == (128, 128)
        if agent == "sac" or HAS_SB3:
            make_agent(agent, 10, 3, "cpu", 0, **hp)


def test_recurrent_ppo_trains_selects_and_reproduces():
    if not HAS_SB3:
        print("  (skipped: sb3-contrib not installed)")
        return
    d, out = _prepared(), Path(tempfile.mkdtemp())
    row = run(_spec("rppo", TINY_RPPO, fallback="cash", fallback_min_score=-1e9), d, d, out, device="cpu",
              verbose=False, eval_splits=("val",))
    rd = out / "models" / "t22" / "rppo_A_net_asset_change_s0"
    mets, _ = backtest_saved(rd, ["val"], device="cpu")
    assert np.isclose(mets["val"]["sharpe"], row["val_sharpe"]) and mets["val"]["num_trades"] == row["val_num_trades"]
    ag = make_agent("rppo", 10, 3, "cpu", 0, **TINY_RPPO)
    pol = agent_policy(ag)
    m = Market(d, d, "val", EnvConfig(state="A"), th.device("cpu"))
    a, b = run_policy(pol, m, EnvConfig(state="A")), run_policy(pol, m, EnvConfig(state="A"))
    assert np.array_equal(a["level"], b["level"])            # the memory is reset between backtests
