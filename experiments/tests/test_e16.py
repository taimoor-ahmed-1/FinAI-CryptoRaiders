"""E16: the stay-flat fallback, hidden test split, saved-run backtests and the always-short baseline."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade.baselines import always_short_policy      # noqa: E402
from rltrade.env import EnvConfig, Market, run_policy  # noqa: E402
from rltrade.metrics import compute_metrics             # noqa: E402
from rltrade.runner import RunSpec, backtest_saved, run  # noqa: E402

TINY_DQN = dict(net_dims=(16, 16), num_envs=4, learning_starts=200, batch_size=32, buffer_size=5_000)


def _prepared(splits=("train", "val"), n=3000, seed=0):
    """Prepared + factor dir with only `splits` on disk (no test files at all)."""
    rng = np.random.default_rng(seed)
    d = Path(tempfile.mkdtemp())
    for k, s in enumerate(splits):
        price = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, n)))
        sig = np.full(n, 3.0, np.float32)
        np.savez(d / f"market_{s}.npz", midpoint=price, spread=np.ones(n),
                 time_ns=(np.arange(n, dtype=np.int64) + k * n) * 60_000_000_000,
                 sentiment_score=sig, risk_score=sig)
        np.save(d / f"factors_{s}.npy", rng.normal(size=(n, 8)).astype(np.float32))
    return d


def _spec(**kw):
    env = EnvConfig(state="A", episode_bars=300, start_skip=10)
    base = dict(experiment="t16", agent="ddqn", seed=0, env=env, agent_hp=TINY_DQN, total_steps=2_000,
                eval_every=1_000, val_chunks=2, min_val_trades=1)
    base.update(kw)
    return RunSpec(**base)


def test_fallback_trades_as_cash_and_test_is_never_loaded():
    d, out = _prepared(), Path(tempfile.mkdtemp())
    row = run(_spec(fallback="cash", fallback_min_score=1e9), d, d, out, device="cpu", verbose=False,
              eval_splits=("val",))
    assert row["fallback_used"] is True
    assert row["val_num_trades"] == 0 and row["val_sharpe"] == 0.0 and row["val_cum_return_pct"] == 0.0
    assert row["ckpt_val_num_trades"] != "" and not any(k.startswith("test_") for k in row)
    m = json.loads((out / "models" / "t16" / "ddqn_A_net_asset_change_s0" / "metrics.json").read_text())
    assert "test" not in m and m["ckpt_val"]["num_trades"] == row["ckpt_val_num_trades"]


def test_no_fallback_when_checkpoint_beats_threshold_and_saved_run_reproduces():
    d, out = _prepared(splits=("train", "val", "test")), Path(tempfile.mkdtemp())
    row = run(_spec(fallback="cash", fallback_min_score=-1e9), d, d, out, device="cpu", verbose=False,
              eval_splits=("val",))
    assert row["fallback_used"] is False and row["ckpt_val_sharpe"] == row["val_sharpe"]
    rd = out / "models" / "t16" / "ddqn_A_net_asset_change_s0"
    mets, _ = backtest_saved(rd, ["val", "test"], device="cpu")
    assert np.isclose(mets["val"]["sharpe"], row["val_sharpe"]) and mets["val"]["num_trades"] == row["val_num_trades"]
    assert mets["test"]["steps"] > 0


def test_saved_fallback_run_stays_flat_on_new_splits():
    d, out = _prepared(splits=("train", "val", "test")), Path(tempfile.mkdtemp())
    run(_spec(fallback="cash", fallback_min_score=1e9), d, d, out, device="cpu", verbose=False, eval_splits=("val",))
    rd = out / "models" / "t16" / "ddqn_A_net_asset_change_s0"
    mets, _ = backtest_saved(rd, ["test"], device="cpu")
    assert mets["test"]["num_trades"] == 0
    raw, _ = backtest_saved(rd, ["val"], device="cpu", use_fallback=False)    # the checkpoint itself
    ckpt = json.loads((rd / "metrics.json").read_text())["ckpt_val"]
    assert raw["val"]["num_trades"] == ckpt["num_trades"] and np.isclose(raw["val"]["sharpe"], ckpt["sharpe"])


def test_default_runner_still_scores_test():
    d, out = _prepared(splits=("train", "val", "test")), Path(tempfile.mkdtemp())
    row = run(_spec(), d, d, out, device="cpu", verbose=False)
    assert "test_sharpe" in row and row["fallback_used"] is False
    assert (out / "models" / "t16" / "ddqn_A_net_asset_change_s0" / "test_backtest.npz").exists()


def test_always_short_enters_once_and_holds():
    d = _prepared(splits=("val",))
    cfg = EnvConfig(state="A")
    bt = run_policy(always_short_policy, Market(d, d, "val", cfg, "cpu"), cfg)
    assert (bt["level"] == -1).all() and compute_metrics(bt, cfg.periods_per_year)["num_trades"] == 1
