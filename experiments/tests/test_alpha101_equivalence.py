"""The vectorised Alpha101 helpers must reproduce the notebook's originals exactly.

Run:  python -m pytest experiments/tests -q      (or: python tests/test_alpha101_equivalence.py)

Three levels:
  1. each helper vs its original on synthetic data with NaNs, ties and plateaus;
  2. decay_linear vs the original minus its look-ahead `bfill`;
  3. all 101 factors end-to-end, computed once with the vectorised helpers and
     once with the originals, on real bars if available (else synthetic).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import alpha101 as A              # noqa: E402
from rltrade import _reference_helpers as R    # noqa: E402

RTOL, ATOL = 1e-9, 1e-12


def _series(n=3000, seed=0, nan_frac=0.02, ties=True):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 100
    if ties:
        x = np.round(x, 0)                       # many exact ties
        x[500:540] = x[500]                      # a flat plateau
    x[rng.random(n) < nan_frac] = np.nan
    x[rng.random(n) < 0.005] = np.inf            # pandas treats +/-inf as missing in
    x[rng.random(n) < 0.005] = -np.inf           # rolling windows - real bars have them
    return pd.Series(x, name="x")


def _eq(a, b, label):
    a = np.asarray(getattr(a, "values", a), dtype=float).ravel()
    b = np.asarray(getattr(b, "values", b), dtype=float).ravel()
    assert a.shape == b.shape, f"{label}: shape {a.shape} vs {b.shape}"
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{label}: NaN positions differ"
    m = ~np.isnan(a)
    assert np.allclose(a[m], b[m], rtol=RTOL, atol=ATOL), \
        f"{label}: max abs diff {np.nanmax(np.abs(a[m] - b[m]))}"


def test_helpers_match_originals():
    for seed in range(3):
        s = _series(seed=seed)
        for w in (3, 5, 10, 20):
            _eq(A.ts_rank(s, w), R.ts_rank(s, w), f"ts_rank w={w}")
            _eq(A.rank(s, w), R.rank(s, w), f"rank w={w}")
            _eq(A.product(s / 100, w), R.product(s / 100, w), f"product w={w}")
            _eq(A.ts_argmax(s, w), R.ts_argmax(s, w), f"ts_argmax w={w}")
            _eq(A.ts_argmin(s, w), R.ts_argmin(s, w), f"ts_argmin w={w}")


def _decay_linear_original_without_bfill(df, period=10):
    """The notebook's decay_linear with only the `bfill` line removed."""
    if df.isnull().values.any():
        df.ffill(inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :].values
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    for row in range(period - 1, df.shape[0]):
        x = df.iloc[row - period + 1: row + 1, :].values
        na_lwma[row, :] = np.dot(x.T, y)
    return pd.DataFrame(na_lwma, index=df.index, columns=["LWMA"])


def test_decay_linear():
    for seed in range(3):
        s = _series(seed=seed).to_frame()
        for p in (5, 10, 16):
            _eq(A.decay_linear(s.copy(), p), _decay_linear_original_without_bfill(s.copy(), p),
                f"decay_linear p={p}")
            clean = s.ffill().fillna(0)        # no gaps -> bfill irrelevant -> must equal notebook
            _eq(A.decay_linear(clean.copy(), p), R.decay_linear(clean.copy(), p),
                f"decay_linear(no gaps) p={p}")
            flag = (s["x"].diff() > 0).to_frame()     # boolean input keeps a boolean output
            out = A.decay_linear(flag.copy(), p)
            assert out.LWMA.dtype == bool
            _eq(out, _decay_linear_original_without_bfill(flag.copy(), p), f"decay_linear(bool) p={p}")


def _bars(n):
    """Real 1-minute bars if the dataset is present, else a synthetic stand-in."""
    for p in [os.environ.get("BTC_CSV", ""),
              "data/BTC_1min_with_sentiment_risk_train.csv",
              "../data/BTC_1min_with_sentiment_risk_train.csv"]:
        if p and Path(p).exists():
            return pd.read_csv(p, nrows=n), "real"
    rng = np.random.default_rng(1)
    mid = 60000 + np.cumsum(rng.normal(0, 20, n))
    df = pd.DataFrame({"midpoint": mid, "spread": np.abs(rng.normal(20, 5, n)),
                       "buys": np.abs(rng.normal(5, 2, n)), "sells": np.abs(rng.normal(5, 2, n)),
                       "bids_notional_3": np.abs(rng.normal(3e8, 2e7, n)),
                       "asks_notional_3": np.abs(rng.normal(3e8, 2e7, n))})
    df["bids_distance_3"] = -(df.spread / 2) / df.midpoint
    df["asks_distance_3"] = (df.spread / 2) / df.midpoint
    return df, "synthetic"


def test_all_101_alphas_end_to_end(n=1200):
    df, kind = _bars(n)
    fast = A.compute_raw_alphas(df.copy(), verbose=False)

    swapped = {}
    names = ["ts_rank", "rank", "product", "ts_argmax", "ts_argmin", "decay_linear"]
    try:
        for nm in names:
            swapped[nm] = getattr(A, nm)
        for nm in names[:-1]:
            setattr(A, nm, getattr(R, nm))
        A.decay_linear = _decay_linear_original_without_bfill
        slow = A.compute_raw_alphas(df.copy(), verbose=False)
    finally:
        for nm, fn in swapped.items():
            setattr(A, nm, fn)

    bad = [i + 1 for i in range(fast.shape[1])
           if not np.allclose(fast[:, i], slow[:, i], rtol=1e-5, atol=1e-6)]
    assert not bad, f"alphas differ ({kind} data): {bad}"
    nonzero = int((np.abs(fast).sum(axis=0) > 0).sum())
    print(f"  all 101 alphas identical on {n} {kind} bars ({nonzero} non-constant)")


if __name__ == "__main__":
    test_helpers_match_originals();   print("helpers ............ OK")
    test_decay_linear();              print("decay_linear ....... OK")
    test_all_101_alphas_end_to_end(); print("101 alphas ......... OK")
