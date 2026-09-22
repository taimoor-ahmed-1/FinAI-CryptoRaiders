"""
Stage 1 - turn the 1-minute dataset into everything later stages consume.

Writes to `<prepared_dir>/`:
    alpha101_{train,val,test}.npy   Alpha101 factors, train-fitted normalisation (float16)
    labels_{train,val,test}.npy     RNN targets, one set per split (the leakage fix)
    market_{train,val,test}.npz     prices + LLM signals for the trading environment
    meta.json                       split boundaries, date ranges, settings used

The split reproduces `2_data_splitter.ipynb` exactly: chronological,
train = rows [0, int(n*0.70)), val = [.., int(n*0.85)), test = the rest.

LLM SIGNAL LOOK-AHEAD FIX
-------------------------
The dataset stores ONE sentiment / risk value per UTC day - the mean over all of
that day's tweets - and repeats it on every 1-minute bar of the same day. The bar
at 00:05 therefore carries the mood of tweets posted up to 23:59 that day: up to
24 hours of future information, in exactly the features the ablation measures.

The dataset itself is not modified. Here, each bar gets the value of the last
COMPLETED day instead (day D's bars see day D-1's aggregate), which is what a live
system could actually know. The same-day values are kept in the market files
under `sameday__<column>` so the size of the leak can be measured, never as the
default. The very first day has no previous day; it gets the train-period mean.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha101
from .labels import WIN_SIZES, labels_for_splits, max_offset

SPLITS = ("train", "val", "test")
SIGNAL_COLS = ["sentiment_score", "risk_score", "sentiment_weighted", "risk_weighted", "tweet_volume"]
REQUIRED = ["system_time", "midpoint", "spread", "buys", "sells",
            "bids_distance_3", "asks_distance_3", "bids_notional_3", "asks_notional_3",
            "sentiment_score", "risk_score"]


def lag_daily_signals(df, cols, train_end):
    """Shift day-level signals so every bar sees only the previous completed UTC day.

    Returns {col: lagged array}. Works for any calendar gaps: each day takes the
    value of the most recent EARLIER day present in the data.
    """
    day = pd.to_datetime(df["system_time"], utc=True).dt.floor("D")
    daily = df[cols].groupby(day.to_numpy()).last()          # value known once the day is complete
    prev = daily.shift(1)                                      # previous available day
    lagged = prev.reindex(day.to_numpy()).to_numpy()
    fill = df[cols].iloc[:train_end].mean().to_numpy()         # first day only: train-period mean
    lagged = np.where(np.isnan(lagged), fill[None, :], lagged)
    return {c: lagged[:, k].astype(np.float32) for k, c in enumerate(cols)}


def split_bounds(n, train_ratio=0.70, val_ratio=0.15):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}


def prepare(csv_path, prepared_dir, train_ratio=0.70, val_ratio=0.15, nrows=None, verbose=True):
    """Run the whole preparation. `nrows` limits the input (smoke tests only)."""
    t0 = time.time()
    out = Path(prepared_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, nrows=nrows)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    df = df.sort_values("system_time", kind="stable").reset_index(drop=True)
    n = len(df)
    bounds = split_bounds(n, train_ratio, val_ratio)
    if verbose:
        print(f"{n:,} rows | " + " | ".join(f"{k} {b - a:,}" for k, (a, b) in bounds.items()))

    # --- Alpha101: computed once on the full series (all operators look backwards) ---
    if verbose:
        print("Computing 101 Alpha101 factors on the full series ...")
    raw = alpha101.compute_raw_alphas(df, verbose=verbose)
    a0, a1 = bounds["train"]
    lo, hi = alpha101.fit_quantiles(raw[a0:a1])            # fitted on TRAIN only
    for s, (a, b) in bounds.items():
        np.save(out / f"alpha101_{s}.npy", alpha101.apply_quantiles(raw[a:b], lo, hi))
    np.savez(out / "alpha101_norm.npz", lo=lo, hi=hi)
    del raw

    # --- RNN labels: one set per split, thresholds fitted on train ---
    mids = {s: df["midpoint"].to_numpy(np.float64)[a:b] for s, (a, b) in bounds.items()}
    labels = labels_for_splits(mids, WIN_SIZES)
    for s in SPLITS:
        np.save(out / f"labels_{s}.npy", labels[s])

    # --- market arrays for the trading environment ---
    ts = pd.to_datetime(df["system_time"], utc=True)
    sig_cols = [c for c in SIGNAL_COLS if c in df.columns]
    lagged = lag_daily_signals(df, sig_cols, bounds["train"][1])
    for s, (a, b) in bounds.items():
        sig = {c: lagged[c][a:b] for c in sig_cols}
        sameday = {f"sameday__{c}": df[c].to_numpy(np.float32)[a:b] for c in sig_cols}
        np.savez(out / f"market_{s}.npz",
                 midpoint=df["midpoint"].to_numpy(np.float64)[a:b],
                 spread=df["spread"].to_numpy(np.float64)[a:b],
                 time_ns=ts.astype("int64").to_numpy()[a:b],
                 **sig, **sameday)

    meta = {
        "source_csv": str(csv_path),
        "rows": n,
        "split_ratio": [train_ratio, val_ratio, round(1 - train_ratio - val_ratio, 6)],
        "splits": {s: {"rows": [a, b],
                       "start": str(ts.iloc[a]), "end": str(ts.iloc[b - 1]),
                       "labelled_rows": int(labels[s].shape[0])}
                   for s, (a, b) in bounds.items()},
        "label_win_sizes": list(WIN_SIZES),
        "label_max_offset": max_offset(WIN_SIZES),
        "label_thresholds": "fitted on train, reused for val/test",
        "alpha101_normalisation": "1%/99% quantile clip fitted on train, mapped to [-1, 1]",
        "signal_columns": sig_cols,
        "signal_alignment": "lagged one UTC day (bar on day D sees day D-1); same-day values "
                            "kept as sameday__<col> for the leak comparison only",
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        for s, m in meta["splits"].items():
            print(f"  {s:<5} {m['start'][:10]} -> {m['end'][:10]}  rows={m['rows'][1]-m['rows'][0]:,}"
                  f"  labelled={m['labelled_rows']:,}")
        print(f"Prepared in {meta['seconds']}s -> {out}")
    return meta
