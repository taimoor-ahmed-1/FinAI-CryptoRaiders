"""E19: factor threshold rules scored in parallel must match one exact backtest per rule."""
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import factor_rule as FR                  # noqa: E402
from rltrade.env import EnvConfig, Market, run_policy  # noqa: E402
from rltrade.metrics import compute_metrics            # noqa: E402
from tests.test_e16 import _prepared                   # noqa: E402


def test_parallel_rules_match_single_backtests():
    d = _prepared(splits=("train", "val"), n=4000)
    cfg = EnvConfig(state="A")
    m = Market(d, d, "val", cfg, "cpu")
    sig = FR.signals(np.load(d / "factors_train.npy"))
    grid = FR.make_grid(sig, quantiles=(0.5, 0.9))
    assert len(grid) == 9 * 2 * 2 * 2
    pick = [0, 5, 13, 22, 47, len(grid) - 1]
    sub = [grid[i] for i in pick]
    got = FR.run_rules(sub, m, cfg)
    for i, g in enumerate(sub):
        pol = FR.rule_policy([g], cfg, "cpu")
        ref = compute_metrics(run_policy(pol, m, cfg, chunks=1), cfg.periods_per_year)
        for key in ("sharpe", "sortino", "cum_return_pct", "max_drawdown_pct", "num_trades", "costs_pct",
                    "long_pct", "short_pct"):
            assert np.isclose(got[key][i], ref[key], rtol=1e-6, atol=1e-6), (g, key, got[key][i], ref[key])
    assert got["num_trades"].max() > 0


def test_rule_logic_and_edge():
    cfg = EnvConfig(state="A")
    g = [{"j": 8, "k": 0.5, "mode": "hold", "min_hold": 0}, {"j": 0, "k": 0.5, "mode": "flat", "min_hold": 0},
         {"j": 0, "k": 0.5, "mode": "flat", "min_hold": 100}]
    pol = FR.rule_policy(g, cfg, "cpu")
    obs = th.zeros(3, 10)
    obs[:, 0] = 1.0                                 # long
    obs[:, 1] = 50 / cfg.holding_norm_bars          # held 50 bars
    obs[:, 2] = 0.1                                 # weak signal
    a = pol(obs)
    assert a.tolist() == [1, 0, 1]                  # hold keeps; flat exits; min_hold keeps
    obs[:, 2:10] = -0.9
    assert pol(obs).tolist() == [0, 0, 1]           # strong short signal: reduce, unless locked
    fwd = FR.forward_returns(np.array([1.0, 1.0, 2.0, 4.0, 4.0]), h=1)
    assert np.allclose(fwd[:3], [1.0, 1.0, 0.0]) and np.isnan(fwd[3:]).all()
    mean, cnt = FR.edge(np.array([-1.0, 1.0, 1.0]), np.array([0.01, 0.02, np.nan]), cuts=[0.0])
    assert np.allclose(mean, [1.0, 2.0]) and cnt.tolist() == [1, 1]
