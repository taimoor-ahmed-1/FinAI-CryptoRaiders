"""
Long-range context inputs (experiment 15): slow, causal summaries of the last 4 hours and the last day.

The GRU reads windows of 256-2048 one-minute bars but predicts trends up to 900 bars ahead, and nothing in its
inputs describes the day's trend or volatility regime. This adds 18 slow inputs to an existing prepared set,
built from data already in it (the midpoint in market_<split>.npz and the scaled intra-minute inputs):

    mid_ret_{240,1440}        log(midpoint / midpoint k bars earlier)
    mid_gap_ema{240,1440}     log(midpoint / its exponential moving average)
    mid_rv_ema{240,1440}      sqrt(EMA of squared 1-minute log returns of the midpoint)
    <input>__ema{240,1440}    EMA of flow_imbalance, log_dollar_volume, close_vwap_gap, rv_1s, ret_1m, amihud_1s

All are causal (row t uses rows <= t). They are computed on the splits joined end to end in time order, as the
other inputs are; the first rows of train, without a full window yet, use what history exists. The new inputs
are clipped to their TRAIN 1 % / 99 % quantiles and mapped to [-1, 1] like the rest (alpha101.fit_quantiles).

    python -m rltrade.long_context <src prepared dir> <out dir>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha101
from .rnn.train import available_splits

SPANS = (240, 1440)
EMA_INPUTS = ("flow_imbalance", "log_dollar_volume", "close_vwap_gap", "rv_1s", "ret_1m", "amihud_1s")


def slow_features(mid, x, names, spans=SPANS):
    """mid: (T,) midpoint; x: (T, F) scaled inputs with column `names`. Returns (T, K) float64 and K names."""
    mid = np.asarray(mid, dtype=np.float64)
    lm = np.log(mid)
    r1 = np.diff(lm, prepend=lm[0])
    cols, out = [], []
    for k in spans:
        prev = np.concatenate([np.full(k, lm[0]), lm[:-k]]) if len(lm) > k else np.full(len(lm), lm[0])
        out.append(lm - prev); cols.append(f"mid_ret_{k}")
    for k in spans:
        ema = pd.Series(mid).ewm(span=k, adjust=False).mean().to_numpy()
        out.append(lm - np.log(ema)); cols.append(f"mid_gap_ema{k}")
    for k in spans:
        out.append(np.sqrt(pd.Series(r1 ** 2).ewm(span=k, adjust=False).mean().to_numpy())); cols.append(f"mid_rv_ema{k}")
    idx = {n: j for j, n in enumerate(names)}
    for n in EMA_INPUTS:
        if n not in idx:
            continue
        for k in spans:
            out.append(pd.Series(x[:, idx[n]].astype(np.float64)).ewm(span=k, adjust=False).mean().to_numpy())
            cols.append(f"{n}__ema{k}")
    return np.stack(out, axis=1), cols


def build(src, dst, spans=SPANS, verbose=True):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    splits = available_splits(src)
    meta = json.loads((src / "meta.json").read_text())
    names = meta["inputs"]
    xs = {s: np.load(src / f"alpha101_{s}.npy") for s in splits}
    mids = {s: np.load(src / f"market_{s}.npz")["midpoint"] for s in splits}
    for s in splits:
        assert len(xs[s]) == len(mids[s]), f"{s}: inputs and market rows differ"
    x_all = np.concatenate([xs[s].astype(np.float32) for s in splits])
    slow, cols = slow_features(np.concatenate([mids[s] for s in splits]), x_all, names, spans)
    n_tr = len(xs["train"])
    lo, hi = alpha101.fit_quantiles(slow[:n_tr])
    slow_n = alpha101.apply_quantiles(slow, lo, hi)
    o = 0
    for s in splits:
        n = len(xs[s])
        np.save(dst / f"alpha101_{s}.npy", np.concatenate([xs[s], slow_n[o:o + n]], axis=1))
        o += n
        for f in (f"labels_{s}.npy", f"market_{s}.npz"):
            shutil.copy(src / f, dst / f)
    np.savez(dst / "long_context_norm.npz", lo=lo, hi=hi)
    if (src / "alpha101_norm.npz").exists():
        shutil.copy(src / "alpha101_norm.npz", dst / "alpha101_norm.npz")
    m = dict(meta, set=meta.get("set", src.name) + "_long", inputs=names + cols, n_inputs=len(names) + len(cols),
             long_context={"spans": list(spans), "columns": cols, "source_set": src.name,
                           "normalisation": "train 1 % / 99 % quantile clip, mapped to [-1, 1]"})
    (dst / "meta.json").write_text(json.dumps(m, indent=1))
    if verbose:
        print(f"[long_context] {src.name} + {len(cols)} slow inputs -> {m['n_inputs']} inputs in {dst}")
    return m


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
