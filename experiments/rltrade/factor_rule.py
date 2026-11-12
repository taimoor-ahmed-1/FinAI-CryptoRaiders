"""
Experiment 19: can the E15 market factors be traded after costs WITHOUT reinforcement learning?

The 8 factors are the GRU's forecasts of the 8 trend labels (labels.py): factor j predicts how the
smoothed price OFFSETS[j] bars ahead compares with the price now (10 ... 900 one-minute bars).
Two checks, both on train and validation only:

1. Edge: the market's forward return over factor j's horizon, by decile of the factor (cut points
   fitted on train). A rule can only profit if the strongest signals precede moves larger than
   the round-trip cost (2 x (fee + slippage) = 0.24 %).
2. Threshold rules, traded in the SAME environment as the agents (TradingEnv, same costs and
   execution, a decision every 5 bars): go long when the signal > k, short when < -k, and
   otherwise either keep the position ("hold") or go flat ("flat"); optionally keep a new
   position for at least `min_hold` bars. k is a train quantile of |signal|.

`run_rules` scores many rules at once: TradingEnv(mode="parallel") trades the whole split once
per copy, one rule per copy, and the metrics are accumulated step by step (the same definitions
as metrics.compute_metrics), so the grid costs about one backtest.
"""
import numpy as np
import torch as th
from scipy import stats

from .env import TradingEnv
from .labels import WIN_SIZES

OFFSETS = tuple(int(x) for x in np.cumsum(WIN_SIZES))      # bars ahead each factor looks: 10 ... 900
SIGNALS = tuple(f"f{j}" for j in range(8)) + ("mean",)     # the 8 factors and their average
SIGNAL_HORIZON = OFFSETS + (300,)                           # the average's nominal horizon
ROUND_TRIP_COST = 2 * (0.001 + 0.0002)


def signals(factors):
    """(n, 8) factors -> (n, 9): the factors and their average, in SIGNALS order."""
    f = np.asarray(factors, np.float32)
    return np.concatenate([f, f.mean(1, keepdims=True)], axis=1)


def forward_returns(price, h, lag=1):
    """Simple return from the execution price of bar t (t + lag) to h bars later; NaN at the end."""
    p = np.asarray(price, np.float64)
    n = len(p)
    out = np.full(n, np.nan)
    if n - lag - h > 0:
        out[: n - lag - h] = p[lag + h:] / p[lag: n - h] - 1.0
    return out


def edge(sig, fwd, cuts):
    """Mean forward return (%) and count in each bin of `sig` defined by `cuts` (len(cuts) + 1 bins)."""
    ok = np.isfinite(fwd)
    b = np.digitize(sig[ok], cuts)
    r = fwd[ok]
    k = len(cuts) + 1
    mean = np.array([100 * r[b == i].mean() if np.any(b == i) else np.nan for i in range(k)])
    return mean, np.bincount(b, minlength=k)


def ic(sig, fwd, h):
    """Spearman correlation of the signal with the forward return, sampled every h bars
    (non-overlapping horizons, so the p-value is not inflated by overlap)."""
    s, r = sig[::h], fwd[::h]
    ok = np.isfinite(r)
    if ok.sum() < 10 or np.std(s[ok]) == 0:
        return float("nan"), float("nan"), int(ok.sum())
    rho, p = stats.spearmanr(s[ok], r[ok])
    return float(rho), float(p), int(ok.sum())


def make_grid(train_signals, quantiles=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99), modes=("hold", "flat")):
    """Every rule: signal x threshold quantile x mode x min_hold (0 or the signal's horizon)."""
    a = np.abs(np.asarray(train_signals))
    grid = []
    for j, name in enumerate(SIGNALS):
        for q in quantiles:
            k = float(np.quantile(a[:, j], q))
            for mode in modes:
                for mh in (0, SIGNAL_HORIZON[j]):
                    grid.append({"signal": name, "j": j, "q": q, "k": k, "mode": mode, "min_hold": mh})
    return grid


def rule_policy(grid, cfg, device):
    """Vectorised policy: copy i follows grid[i]. Observations are TradingEnv state A rows."""
    j = th.tensor([g["j"] for g in grid], device=device)
    k = th.tensor([g["k"] for g in grid], device=device, dtype=th.float32)
    hold = th.tensor([g["mode"] == "hold" for g in grid], device=device)
    min_hold = th.tensor([g["min_hold"] for g in grid], device=device, dtype=th.float32)

    def policy(obs):
        feat = obs[:, 2:10]
        f = th.cat([feat, feat.mean(1, keepdim=True)], dim=1).gather(1, j[:, None]).squeeze(1)
        level = th.round(obs[:, 0] * cfg.max_position).long()
        held_bars = obs[:, 1] * cfg.holding_norm_bars
        keep = th.where(hold, level, th.zeros_like(level))
        tgt = th.where(f > k, th.ones_like(level), th.where(f < -k, -th.ones_like(level), keep))
        locked = (level != 0) & (held_bars < min_hold - 1e-3)
        tgt = th.where(locked, level, tgt)
        return 1 + th.sign(tgt - level)
    return policy


@th.no_grad()
def run_rules(grid, market, cfg, device="cpu"):
    """Trade every rule in `grid` over the whole split (one exact pass each).

    Returns a dict of arrays (one entry per rule) with the metrics.compute_metrics definitions:
    sharpe, sortino, cum_return_pct, max_drawdown_pct, num_trades, costs_pct, long_pct,
    short_pct, exposure_pct, plus ruined (equity reached 0 at some point).
    """
    n = len(grid)
    env = TradingEnv(market, cfg, num_envs=n, mode="parallel")
    policy = rule_policy(grid, cfg, env.dev)
    obs = env.reset()
    T = env.ep_steps
    z = lambda: th.zeros(n, dtype=th.float64, device=env.dev)   # noqa: E731
    s1, s2, dn, costs, lng, sht, trades, mdd = z(), z(), z(), z(), z(), z(), z(), z()
    eq, peak = z() + 1.0, z() + 1.0
    ruined = th.zeros(n, dtype=th.bool, device=env.dev)
    prev = th.zeros(n, dtype=th.long, device=env.dev)
    for _ in range(T):
        obs, _, _, info = env.step(policy(obs))
        r = info["eq_next"] / info["eq_prev"] - 1.0
        s1 += r
        s2 += r * r
        dn += th.clamp(r, max=0.0) ** 2
        eq = eq * (1.0 + r)
        ruined |= eq <= 0
        peak = th.maximum(peak, eq)
        mdd = th.maximum(mdd, 1.0 - eq / peak)
        lv = info["level"]
        trades += (lv != prev).double()
        prev = lv
        costs += info["cost"] / info["eq_prev"]
        lng += (lv > 0).double()
        sht += (lv < 0).double()
    ppy = cfg.periods_per_year
    mean = s1 / T
    sd = th.sqrt(th.clamp((s2 - s1 * s1 / T) / (T - 1), min=0.0))
    down = th.sqrt(dn / T)
    out = {"sharpe": th.where(sd > 0, mean / sd * np.sqrt(ppy), th.zeros_like(sd)),
           "sortino": th.where(down > 0, mean / down * np.sqrt(ppy), th.zeros_like(down)),
           "cum_return_pct": 100 * (eq - 1.0), "max_drawdown_pct": 100 * th.clamp(mdd, max=1.0),
           "num_trades": trades, "costs_pct": 100 * costs, "long_pct": 100 * lng / T,
           "short_pct": 100 * sht / T, "exposure_pct": 100 * (lng + sht) / T, "ruined": ruined}
    return {k: v.cpu().numpy() for k, v in out.items()}
