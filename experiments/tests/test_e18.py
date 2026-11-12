"""E18: traces of saved runs and the diagnosis statistics."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import diagnose as D                       # noqa: E402
from rltrade.runner import backtest_saved, run           # noqa: E402
from tests.test_e16 import _prepared, _spec               # noqa: E402


def test_trace_reproduces_saved_backtest_and_attribution_adds_up():
    d, out = _prepared(splits=("train", "val", "test")), Path(tempfile.mkdtemp())
    run(_spec(fallback="cash", fallback_min_score=-1e9), d, d, out, device="cpu", verbose=False, eval_splits=("val",))
    rd = out / "models" / "t16" / "ddqn_A_net_asset_change_s0"
    _, bts = backtest_saved(rd, ["test"], device="cpu")
    tr = D.trace(rd, "test", d, d, device="cpu")
    assert np.allclose(tr["returns"], bts["test"]["returns"]) and np.array_equal(tr["level"], bts["test"]["level"])
    assert tr["q"].shape == (len(tr["level"]), 3) and len(tr["held"]) == len(tr["level"])
    assert (D.q_gap(tr) >= 0).all()
    a = D.attribution(tr)
    assert np.isclose(a["exposure_pct"] + a["timing_pct"], a["gross_pct"])
    assert np.isclose(a["from_long_pct"] + a["from_short_pct"], a["gross_pct"])
    assert abs(a["residual_pct"]) < 0.05 * (abs(a["gross_pct"]) + a["costs_pct"] + 1e-3) + 1e-6
    # a baseline has no Q-values and buy and hold's market exposure is ~1
    rb = out / "models" / "t16"
    run(_spec(agent="buy_and_hold"), d, d, out, device="cpu", verbose=False, eval_splits=("val",))
    tb = D.trace(rb / "buy_and_hold_A_net_asset_change_s0", "test", d, d, device="cpu")
    assert tb["q"].shape[1] == 0 and len(D.q_gap(tb)) == 0
    assert abs(D.attribution(tb)["beta"] - 1) < 0.01


def test_statistics():
    assert D.spells([0, 1, 1, -1, -1, -1, 0]) == [(0, 1), (1, 2), (-1, 3), (0, 1)]
    tr = {"level": np.array([0, 1, 1, -1, -1, -1, 0] * 288), "cost": np.zeros(7 * 288),
          "held": np.zeros(7 * 288), "mret": np.zeros(7 * 288), "returns": np.zeros(7 * 288)}
    b = D.behaviour(tr)
    assert b["positions"] == 2 * 288 and np.isclose(b["long_pct"] + b["short_pct"] + b["flat_pct"], 100)
    static = {"level": np.ones(288 * 5), "mret": np.random.default_rng(0).normal(0, 1e-3, 288 * 5)}
    assert np.isnan(D.daily_timing(static)["ic"]) and D.daily_timing(static)["days"] == 5
    ag = D.agreement(np.array([[1, 1, -1, 0], [1, -1, -1, 0], [1, 1, 1, 0]]))
    assert np.isclose(ag["pairwise_same_pct"], 100 * np.mean([3 / 4, 3 / 4, 2 / 4]))
    assert np.isclose(ag["majority_long_pct"], 50.0)
    r = np.random.default_rng(1).normal(1e-4, 1e-3, 288 * 30)
    bs = D.bootstrap_sharpe(r, r, n_boot=200)
    assert bs["beats_ref_pct"] == 0.0 and bs["sharpe_p5"] <= bs["sharpe_p95"]
    rows, any_out = D.factor_shift(np.random.default_rng(2).normal(size=(5000, 2)),
                                   np.random.default_rng(3).normal(3, 1, size=(5000, 2)))
    assert all(r["ks"] > 0.5 and r["mean_shift_sd"] > 2 for r in rows) and any_out > 50
