"""
Performance metrics (brief section 4.1), all computed from a backtest recorded by
`env.run_policy`, i.e. NET of transaction costs.

Conventions (state these in the thesis):
    - step returns are simple returns between consecutive decisions;
    - Sharpe and Sortino are annualised with sqrt(periods per year), crypto trading
      365 x 24 h, risk-free rate 0;
    - max drawdown is reported as a positive percentage;
    - a "trade" is any change of position level; a "round trip" runs from opening
      a position (from flat, or by flipping sides) until it is closed or flipped;
      win rate = share of round trips with positive net P&L (costs included);
      a position still open at the end counts, marked to market;
    - turnover = total traded notional / average equity.
"""
import numpy as np


def max_drawdown(equity):
    eq = np.asarray(equity, dtype=np.float64)
    return float(max(0.0, -(eq / np.maximum.accumulate(eq) - 1.0).min()))


def drawdown_curve(equity):
    eq = np.asarray(equity, dtype=np.float64)
    return eq / np.maximum.accumulate(eq) - 1.0


def round_trips(level, equity):
    """Net P&L of every round trip, from the level after each decision and equity marks."""
    level = np.asarray(level)
    prev = np.concatenate([[0], level[:-1]])
    opens = (level != 0) & (np.sign(level) != np.sign(prev))
    closes = (prev != 0) & (np.sign(level) != np.sign(prev))
    pnl, start = [], None
    for t in range(len(level)):
        if closes[t] and start is not None:
            pnl.append(equity[t + 1] / equity[start] - 1.0)
            start = None
        if opens[t]:
            start = t
    if start is not None:
        pnl.append(equity[-1] / equity[start] - 1.0)
    return np.asarray(pnl)


def compute_metrics(bt, periods_per_year):
    """bt: dict from env.run_policy. Returns a flat dict of floats."""
    r = np.asarray(bt["returns"], dtype=np.float64)
    eq = np.asarray(bt["equity"], dtype=np.float64)
    level = np.asarray(bt["level"])
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else 0.0
    years = len(r) / periods_per_year
    total = eq[-1] - 1.0
    mdd = max_drawdown(eq)
    trips = round_trips(level, eq)
    changes = int(np.count_nonzero(np.diff(np.concatenate([[0], level]))))
    return {
        "cum_return_pct": 100.0 * total,
        "annual_return_pct": 100.0 * (eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else float("nan"),
        "sharpe": float(r.mean() / sd * ann) if sd > 0 else 0.0,
        "sortino": float(r.mean() / downside * ann) if downside > 0 else 0.0,
        "max_drawdown_pct": 100.0 * mdd,
        "romad": float(total / mdd) if mdd > 0 else 0.0,
        "win_rate_pct": 100.0 * float((trips > 0).mean()) if len(trips) else float("nan"),
        "round_trips": int(len(trips)),
        "num_trades": changes,
        "turnover": float(np.sum(bt["traded"]) / np.mean(eq)),
        "exposure_pct": 100.0 * float(np.mean(level != 0)),
        "long_pct": 100.0 * float(np.mean(level > 0)),
        "short_pct": 100.0 * float(np.mean(level < 0)),
        "costs_pct": 100.0 * float(np.sum(bt["cost"])),
        "steps": int(len(r)),
    }


def mean_std(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
