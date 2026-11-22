"""
Experiment 23, step 2: extra inputs for the factor model (GRU) that the prepared sets do not contain.

E15's inputs (fixed_plus / fixed_plus_long) are built from 1-second spot klines only: alpha101, intra-minute
statistics and their slow EMAs. An IC scan on train + validation (E23) found small but stable signals in data that
never reached the model:

    order book       book_imb_l1, book_imb_all (5 levels, 1-5 % from mid; futures book), their EMAs, depth change
    taker flow       (taker buys - sells) / volume over 60 / 300 / 1440 minutes
    derivatives      funding rate and its 1-day change, open-interest change (1 h, 1 day) and OI change x sign of the
                     1-day return, top-trader and global long/short ratios (log) and their 1-day change, taker
                     long/short volume ratio (Binance USD-M futures metrics, build_futures_data.py)
    ETH              ETH returns (1 h, 5 h, 1 day), ETH minus BTC return (1 h, 1 day), ETH taker flow (1 h)
    sentiment        1-day change of the (lagged, causal) daily LLM sentiment / risk scores and log tweet volume

All features are causal: row t uses data up to bar t only. Derivatives snapshots are aligned so a snapshot taken
at time s is used from bar s on (build_futures_data.py); the daily sentiment scores are the prepared set's lagged
columns (the previous day's tweets).

Two stages:
    raw(...)    on the machine that has the raw CSVs: writes extra_<split>.npy (float32, unscaled) aligned row by
                row with the prepared set's market_<split>.npz, plus extra.json with the column names
    build(...)  wherever the prepared set is (Colab): forward-fills gaps, fills what is left with the train median,
                clips to the train 1 % / 99 % quantiles and maps to [-1, 1] (as alpha101 / long_context), and
                appends the columns to alpha101_<split>.npy of a source set -> a new prepared set

    python -m rltrade.enrich raw <btc_1m_csv> <futures_extra_csv> <prepared_dir> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha101
from .rnn.train import available_splits

SENTIMENT_COLS = ("sentiment_score", "risk_score", "sentiment_weighted", "risk_weighted", "tweet_volume")
DAY = 1440


def extra_features(d):
    """d: DataFrame of consecutive 1-minute bars with midpoint, buys, sells, bids_notional_0..4, asks_notional_0..4,
    oi_btc, top_ls_accounts, top_ls_positions, global_ls_accounts, taker_ls_vol_ratio, funding_rate, eth_mid,
    eth_volume, eth_taker_buy and the lagged daily SENTIMENT_COLS. Returns a float64 DataFrame of causal features."""
    F = pd.DataFrame(index=d.index)
    lp = np.log(d["midpoint"].astype(np.float64))
    bid = sum(d[f"bids_notional_{i}"] for i in range(5))
    ask = sum(d[f"asks_notional_{i}"] for i in range(5))
    F["book_imb_l1"] = (d.bids_notional_0 - d.asks_notional_0) / (d.bids_notional_0 + d.asks_notional_0)
    imb = (bid - ask) / (bid + ask)
    F["book_imb_all"] = imb
    for k in (60, DAY):
        F[f"book_imb_all_ema{k}"] = imb.ewm(span=k, adjust=False).mean()
    ldep = np.log(bid + ask)
    for k in (60, DAY):
        F[f"depth_change_{k}"] = ldep - ldep.shift(k)
    flow, vol = d.buys - d.sells, d.buys + d.sells
    for k in (60, 300, DAY):
        F[f"taker_flow_{k}"] = flow.rolling(k, min_periods=1).sum() / vol.rolling(k, min_periods=1).sum()
    F["funding_rate"] = d.funding_rate
    F["funding_change_1d"] = d.funding_rate - d.funding_rate.shift(DAY)
    loi = np.log(d.oi_btc)
    for k in (60, DAY):
        F[f"oi_change_{k}"] = loi - loi.shift(k)
    F["oi_x_price_1d"] = np.sign(lp - lp.shift(DAY)) * F[f"oi_change_{DAY}"]
    for c in ("top_ls_accounts", "top_ls_positions", "global_ls_accounts"):
        lc = np.log(d[c])
        F[f"{c}_log"] = lc
        F[f"{c}_change_1d"] = lc - lc.shift(DAY)
    F["taker_ls_ratio_ema60"] = np.log(d.taker_ls_vol_ratio).ewm(span=60, adjust=False).mean()
    le = np.log(d.eth_mid.astype(np.float64))
    for k in (60, 300, DAY):
        F[f"eth_ret_{k}"] = le - le.shift(k)
    for k in (60, DAY):
        F[f"eth_minus_btc_{k}"] = (le - le.shift(k)) - (lp - lp.shift(k))
    F["eth_taker_flow_60"] = ((2 * d.eth_taker_buy - d.eth_volume).rolling(60, min_periods=1).sum()
                              / d.eth_volume.rolling(60, min_periods=1).sum())
    for c in SENTIMENT_COLS:
        x = np.log1p(d[c].clip(lower=0)) if c == "tweet_volume" else d[c]
        F[f"{c}_change_1d"] = x - x.shift(DAY)
    return F.replace([np.inf, -np.inf], np.nan)


BTC_COLS = ["system_time", "midpoint", "buys", "sells"] + [f"{s}_notional_{i}" for s in ("bids", "asks") for i in range(5)]


def raw(btc_csv, futures_csv, prepared_dir, out_dir):
    """Writes extra_<split>.npy (unscaled float32) row-aligned with prepared_dir/market_<split>.npz."""
    prepared_dir, out = Path(prepared_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    btc = pd.read_csv(btc_csv, usecols=BTC_COLS)
    btc.index = pd.to_datetime(btc.pop("system_time"), utc=True)
    fx = pd.read_csv(futures_csv)
    fx.index = pd.to_datetime(fx.pop("system_time"), utc=True)
    d = btc.join(fx, how="left")
    splits = available_splits(prepared_dir)
    mk = {s: np.load(prepared_dir / f"market_{s}.npz") for s in splits}
    t = {s: pd.to_datetime(mk[s]["time_ns"], unit="ns", utc=True) for s in splits}
    sent = pd.DataFrame({c: np.concatenate([mk[s][c] for s in splits]) for c in SENTIMENT_COLS},
                        index=pd.DatetimeIndex(np.concatenate([t[s] for s in splits])))
    d = d.join(sent, how="left")
    d[list(SENTIMENT_COLS)] = d[list(SENTIMENT_COLS)].ffill()      # embargo days between splits: carry the last value
    F = extra_features(d)
    for s in splits:
        x = F.reindex(t[s])
        assert len(x) == len(mk[s]["midpoint"]), s
        assert np.allclose(d["midpoint"].reindex(t[s]).to_numpy(), mk[s]["midpoint"], rtol=1e-6, equal_nan=True), \
            f"{s}: CSV midpoint does not match the prepared set - wrong file or alignment"
        np.save(out / f"extra_{s}.npy", x.to_numpy(np.float32))
    info = {"columns": list(F.columns), "splits": {s: int(len(t[s])) for s in splits}, "source": [str(btc_csv), str(futures_csv)],
            "note": "unscaled; rltrade.enrich.build scales with train statistics"}
    (out / "extra.json").write_text(json.dumps(info, indent=1))
    return info


def build(src, extra_dir, dst, verbose=True):
    """Appends the scaled extra columns to src's alpha101_<split>.npy -> a new prepared set in dst."""
    src, extra_dir, dst = Path(src), Path(extra_dir), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    splits = available_splits(src)
    meta = json.loads((src / "meta.json").read_text())
    info = json.loads((extra_dir / "extra.json").read_text())
    ex = {s: np.load(extra_dir / f"extra_{s}.npy").astype(np.float64) for s in splits}
    xs = {s: np.load(src / f"alpha101_{s}.npy") for s in splits}
    for s in splits:
        assert len(ex[s]) == len(xs[s]), f"{s}: extra inputs and prepared rows differ"
    allx = pd.DataFrame(np.concatenate([ex[s] for s in splits])).ffill().to_numpy()
    n_tr = len(ex["train"])
    med = np.nanmedian(allx[:n_tr], axis=0)
    allx = np.where(np.isfinite(allx), allx, med)
    lo, hi = alpha101.fit_quantiles(allx[:n_tr])
    xn = alpha101.apply_quantiles(allx, lo, hi)
    o = 0
    for s in splits:
        n = len(xs[s])
        np.save(dst / f"alpha101_{s}.npy", np.concatenate([xs[s], xn[o:o + n]], axis=1))
        o += n
        for f in (f"labels_{s}.npy", f"market_{s}.npz"):
            shutil.copy(src / f, dst / f)
    np.savez(dst / "enrich_norm.npz", lo=lo, hi=hi, median=med)
    for f in ("alpha101_norm.npz", "long_context_norm.npz"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    cols = info["columns"]
    m = dict(meta, set=meta.get("set", src.name) + "_e23", inputs=meta["inputs"] + cols,
             n_inputs=len(meta["inputs"]) + len(cols),
             enrich={"columns": cols, "source_set": src.name,
                     "normalisation": "ffill, train median for leading gaps, train 1 % / 99 % quantile clip, [-1, 1]"})
    (dst / "meta.json").write_text(json.dumps(m, indent=1))
    if verbose:
        print(f"[enrich] {src.name} + {len(cols)} extra inputs -> {m['n_inputs']} inputs in {dst}")
    return m


if __name__ == "__main__":
    if sys.argv[1] == "raw":
        print(raw(*sys.argv[2:6]))
    else:
        raise SystemExit("usage: python -m rltrade.enrich raw <btc_1m_csv> <futures_extra_csv> <prepared_dir> <out_dir>")
