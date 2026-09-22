"""Behaviour of the reward functions on hand-made equity paths."""
import sys
from pathlib import Path

import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade.rewards import make_reward   # noqa: E402

L0 = th.zeros(1, dtype=th.float64)


def _path(rets):
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return th.tensor(eq, dtype=th.float64)


def _rewards(name, rets, **kw):
    rf = make_reward(name, 1, "cpu", **kw)
    eq = _path(rets)
    return th.stack([rf(eq[i:i + 1], eq[i + 1:i + 2], L0, L0) for i in range(len(rets))]).squeeze(1)


def test_net_asset_change_is_profit_times_scale():
    r = _rewards("net_asset_change", [0.01, -0.02], scale=100.0)
    assert th.allclose(r, th.tensor([1.0, -1.01 * 2], dtype=th.float64), atol=1e-12)


def test_drawdown_penalty_only_when_drawdown_deepens():
    rets = [0.02, -0.01, -0.01, 0.015, 0.02]
    base = _rewards("net_asset_change", rets, scale=100.0)
    dd = _rewards("drawdown_penalized", rets, scale=100.0, lam=1.0)
    assert th.isclose(dd[0], base[0])                    # new high: no penalty
    assert dd[1] < base[1] and dd[2] < base[2]           # falling below the peak: penalised
    assert th.isclose(dd[3], base[3])                    # recovering: no penalty


def test_diff_sharpe_is_finite_and_signed():
    flat = _rewards("diff_sharpe", [0.0] * 50)
    assert th.isfinite(flat).all() and flat.abs().max() < 1e-9
    after_flat = _rewards("diff_sharpe", [0.0] * 200 + [0.001])      # no blow-up after a long flat spell
    assert th.isfinite(after_flat).all() and 0 < after_flat[-1] <= 5.0
    ups = _rewards("diff_sharpe", [0.001] * 20)
    downs = _rewards("diff_sharpe", [-0.001] * 20)
    assert ups[0] > 0 and downs[0] < 0


def test_legacy_shaped_pays_for_trading():
    rf = make_reward("legacy_shaped", 1, "cpu", scale=100.0)
    one = th.ones(1, dtype=th.float64)
    assert float(rf(one, one, L0, one)) > 0               # no profit, but a position change is rewarded


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name:<48} OK")
