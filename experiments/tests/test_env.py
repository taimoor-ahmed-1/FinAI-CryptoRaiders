"""Accounting checks for the trading environment, on tiny synthetic markets."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade.baselines import buy_and_hold_policy       # noqa: E402
from rltrade.env import EnvConfig, Market, TradingEnv, run_policy   # noqa: E402
from rltrade.metrics import compute_metrics              # noqa: E402


def _market(price, sentiment=None, split="test"):
    d = Path(tempfile.mkdtemp())
    n = len(price)
    s = np.full(n, 3.0, np.float32) if sentiment is None else np.asarray(sentiment, np.float32)
    np.savez(d / f"market_{split}.npz", midpoint=np.asarray(price, float), spread=np.ones(n),
             time_ns=np.arange(n, dtype=np.int64) * 60_000_000_000,
             sentiment_score=s, risk_score=s, sameday__sentiment_score=s, sameday__risk_score=s)
    np.save(d / f"factors_{split}.npy", np.zeros((n, 8), np.float32))
    return d


def _cfg(**kw):
    base = dict(step_gap=1, exec_lag=1, fee_rate=0.001, slippage=0.0002)
    base.update(kw)
    return EnvConfig(**base)


def test_buy_and_hold_closed_form():
    rng = np.random.default_rng(0)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    d, cfg = _market(price), _cfg()
    bt = run_policy(buy_and_hold_policy, Market(d, d, "test", cfg, "cpu"), cfg)
    T = len(bt["returns"])
    # fully invested after paying the entry cost: units = 1 / (p0 * (1 + c))
    expected = price[T * cfg.step_gap + cfg.exec_lag] / price[cfg.exec_lag] / (1 + cfg.cost_rate)
    assert np.isclose(bt["equity"][-1], expected, rtol=1e-12), (bt["equity"][-1], expected)
    assert compute_metrics(bt, cfg.periods_per_year)["num_trades"] == 1


def test_round_trip_costs_at_flat_price():
    d, cfg = _market(np.full(50, 100.0)), _cfg()
    env = TradingEnv(Market(d, d, "test", cfg, "cpu"), cfg, num_envs=1, mode="eval")
    env.reset()
    c = cfg.cost_rate
    _, _, _, i1 = env.step(th.tensor([2]))           # open long
    assert np.isclose(float(i1["eq_next"]), 1 / (1 + c), rtol=1e-12)
    assert abs(float(env.cash[0])) < 1e-12                           # fully long, no leverage
    _, _, _, i2 = env.step(th.tensor([0]))           # close
    assert np.isclose(float(i2["eq_next"]), (1 - c) / (1 + c), rtol=1e-12)


def test_short_gains_when_price_falls():
    d, cfg = _market(np.linspace(100, 90, 30)), _cfg(fee_rate=0.0, slippage=0.0)
    env = TradingEnv(Market(d, d, "test", cfg, "cpu"), cfg, num_envs=1, mode="eval")
    env.reset()
    env.step(th.tensor([0]))                          # 0 -> level -1 (short)
    _, r, _, info = env.step(th.tensor([1]))          # hold
    assert float(info["eq_next"]) > float(info["eq_prev"]) and float(r) > 0


def test_no_shorts_when_disabled():
    d, cfg = _market(np.linspace(100, 90, 30)), _cfg(allow_short=False)
    env = TradingEnv(Market(d, d, "test", cfg, "cpu"), cfg, num_envs=1, mode="eval")
    env.reset()
    _, _, _, info = env.step(th.tensor([0]))
    assert int(info["level"]) == 0 and float(info["cost"]) == 0.0


def test_trade_executes_at_next_bar_and_state_is_current():
    price = np.array([100, 200, 300, 400, 500, 600, 700, 800], float)
    sent = np.arange(len(price), dtype=np.float32) + 1              # distinct per bar
    d, cfg = _market(price, sent), _cfg(fee_rate=0.0, slippage=0.0, state="B")
    env = TradingEnv(Market(d, d, "test", cfg, "cpu"), cfg, num_envs=1, mode="eval")
    obs = env.reset()
    assert np.isclose(float(obs[0, -1]), (1 - 1) / 4)              # bar 0's signal
    obs, _, _, info = env.step(th.tensor([2]))                        # buy, filled at bar 1 = 200
    assert np.isclose(float(env.units[0]), 1 / 200)                 # no costs in this test
    assert np.isclose(float(obs[0, -1]), (2 - 1) / 4)              # state moved to bar 1
    assert np.isclose(float(info["eq_next"]), 300 / 200)            # marked at bar 2


def test_cash_baseline_never_trades():
    from rltrade.baselines import cash_policy
    rng = np.random.default_rng(1)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    d, cfg = _market(price), _cfg()
    bt = run_policy(cash_policy, Market(d, d, "test", cfg, "cpu"), cfg)
    m = compute_metrics(bt, cfg.periods_per_year)
    assert m["num_trades"] == 0 and m["cum_return_pct"] == 0.0 and m["max_drawdown_pct"] == 0.0


def test_never_trading_checkpoint_is_not_eligible():
    from rltrade.runner import INVALID_SCORE, selection_score
    flat = {"sharpe": 0.0, "num_trades": 0}
    losing_trader = {"sharpe": -1.5, "num_trades": 12}
    assert selection_score(flat) == (INVALID_SCORE, False)
    assert selection_score(losing_trader) == (-1.5, True)
    assert selection_score(losing_trader)[0] > selection_score(flat)[0]   # trading always outranks doing nothing
    one_bet = {"sharpe": 1.8, "num_trades": 1}                           # short once and hold
    assert selection_score(one_bet, min_trades=5) == (INVALID_SCORE, False)
    assert selection_score(losing_trader, min_trades=5) == (-1.5, True)
    from rltrade.runner import RunSpec
    assert RunSpec("x", "ppo").min_val_trades == 5


def test_state_dims():
    for s, dim in (("A", 10), ("B", 11), ("C", 11), ("D", 12)):
        assert EnvConfig(state=s).state_dim == dim


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name:<52} OK")
