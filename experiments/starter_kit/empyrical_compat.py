"""Stand-in for empyrical 0.5.5 (its setup script fails on Python 3.12).
Only the two functions metrics.py uses, computed exactly as in empyrical 0.5.5 (stats.py)."""
import numpy as np
import pandas as pd


def cum_returns(returns, starting_value=0):
    r = np.asanyarray(returns, dtype="float64").copy()
    r[np.isnan(r)] = 0.0
    out = np.cumprod(r + 1.0, axis=0)
    out = out - 1.0 if starting_value == 0 else out * starting_value
    return pd.Series(out, index=returns.index) if isinstance(returns, pd.Series) else out


def max_drawdown(returns):
    r = np.asanyarray(returns, dtype="float64")
    if len(r) < 1:
        return np.nan
    cumulative = np.empty((r.shape[0] + 1,) + r.shape[1:], dtype="float64")
    cumulative[0] = start = 100
    cumulative[1:] = cum_returns(r, starting_value=start)
    max_return = np.fmax.accumulate(cumulative, axis=0)
    return np.nanmin((cumulative - max_return) / max_return, axis=0)
