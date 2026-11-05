"""
Experiment 13 - a daily dataset built the way Tian, Yang, Wei & Wang (2026) build theirs ("Interval-valued
forecasting and trading strategy for Bitcoin", Pacific-Basin Finance Journal 103352), on the thesis data
(Binance BTC 2024-01 -> 2026-06 + the scraped tweets), for the RL pipeline.

    python -m rltrade.paper_daily vader    --data <dir>  --tweets bitcoin_tweets_scored.csv
    python -m rltrade.paper_daily build    --data <dir>  --btc BTC_1m_2024-01-01_2026-06-06.csv --workers 6
    python -m rltrade.paper_daily precheck --data <dir>

The paper, step by step, and what is done here
    1  interval BTC price        weekly [min, max] (Investing.com)  ->  daily [low, high] of the 1-minute
                                 Binance bars (bar high / low = midpoint +- spread / 2)
    2  interval market data      gold, WTI, U.S. dollar index, S&P 500 as weekly [min, max] (Investing.com)
                                 ->  daily [Low, High] (Yahoo Finance: GC=F, CL=F, DX-Y.NYB, ^GSPC); days
                                 without trading keep the last trading day's interval
    3  attention                 Google Trends 'Bitcoin', [min, max] of the index within the week  ->
                                 [min, max] of the daily index over the last 7 days. Daily values come in
                                 200-day windows (Google rescales each to 100); they are chained on their
                                 overlaps, anchored on the first window, so a value never depends on later
                                 windows. One extra day of lag: Google publishes a day's value late.
    4  sentiment                 VADER compound of CoinDesk headline + summary, weekly [q25, q75]  ->
                                 daily [q25, q75] of the tweets' scores, with VADER (tweet text, URLs
                                 removed) AND with our DeepSeek sentiment scores (1-5). Two deviations:
                                 VADER ignores tweets scored exactly 0, DeepSeek uses continuity-corrected
                                 quartiles of its integer scores - as written, the intervals are [0, x] and
                                 [3, 4] on most days (daily_sentiment)
    5  NA-MEMD                   all 7 interval series (14 channels) decomposed jointly with noise-assisted
                                 multivariate EMD into IMFs. The paper decomposes the whole 2013-25 sample at
                                 once, so every point uses the future. Here the decomposition is CAUSAL: it
                                 is recomputed every day on the data up to that day (expanding window, at
                                 least MIN_DAYS), and only that day's value is kept.
    6  sample entropy grouping   the paper's two-step rule, per variable: IMFs with lower sample entropy than
                                 the original series -> low frequency (with the residue); of the rest, IMFs
                                 above the entropy of their sum -> high, the others -> medium. Deviation:
                                 daily tweet sentiment is so noisy that every IMF is more regular than the
                                 series, so the rule puts ALL of it in low and the high band (the paper's
                                 sentiment channel) is empty on every day; the same happens to BTC on
                                 ~10-25% of days. When the high band is empty, IMF 0 (the fastest) becomes it.
    7  events                    interval event dummies, the paper's Table 4 windows that overlap our data:
                                 halving 2023-10 -> 2024-06, Trump policy cycle 2024-06 -> 2025-05. Kept in
                                 the dataset, NOT in the RL state: they were chosen with hindsight and are 0
                                 on all validation and test days.
    8  models                    IVAR (low: BTC + macro), IEA (medium: BTC + Google Trends + events), IMLP
                                 (high: BTC + sentiment)  ->  the RL agent; it receives the same inputs per
                                 band (features below)
    9  trading                   trade only when the point forecast lies inside the forecast interval  ->
                                 state H: no opening / enlarging while the sentiment interval is wider than
                                 its train median (as state F in experiment 12)

Features (fixed before any check; each day's value uses data up to the END of that day; the RL bars of
day D+1 receive day D's row - prepare's one-day lag)
    btc_trend_gap    BTC centre / centre of its low band - 1          (price vs long-run trend)
    btc_cycle        medium-band centre / BTC centre                  (event-scale cycle)
    btc_noise        high-band centre / BTC centre
    btc_noise_width  high-band width / BTC centre                     (short-term range)
    gold_trend, wti_trend, dxy_trend, sp500_trend
                     5-day change of the variable's low-band centre, relative   (IVAR inputs)
    gt_cycle, gt_cycle_width
                     Google Trends medium band centre / width, relative to its low band   (IEA input)
    sent_noise, sent_noise_width
                     sentiment high band centre / width (scorer: vader or deepseek)       (IMLP input)
Scaling as in experiment 12: train mean, train 99th percentile of |x - mean|, clipped to [-1, 1]; days
before the first decomposition (MIN_DAYS) get 0.
"""
import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import spearmanr

START, END = "2024-01-01", "2026-06-06"
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15             # prepare.split_bounds on 1-minute rows = whole days here
VARIABLES = ("btc", "gold", "wti", "dxy", "sp500", "gt", "sent")
MACRO = ("gold", "wti", "dxy", "sp500")
SCORERS = ("vader", "deepseek")
EVENTS = {"event_halving": ("2023-10-01", "2024-06-30"), "event_trump": ("2024-06-01", "2025-05-31")}
GT_WINDOW = 7
MIN_DAYS = 120                                  # first causal decomposition after 120 days of data
N_DIRS, N_SIFT, MAX_IMFS = 64, 10, 10
NOISE_CHANNELS, NOISE_STD = 2, 0.2              # NA-MEMD: extra white-noise channels (z-scored signals)
SE_M, SE_R = 2, 0.2
FEATURES = ("btc_trend_gap", "btc_cycle", "btc_noise", "btc_noise_width", "gold_trend", "wti_trend",
            "dxy_trend", "sp500_trend", "gt_cycle", "gt_cycle_width", "sent_noise", "sent_noise_width")
_URL = re.compile(r"https?://\S+")


def split_days(n):
    a = int(n * TRAIN_RATIO)
    return {"train": (0, a), "val": (a, int(n * (TRAIN_RATIO + VAL_RATIO))), "test": (int(n * (TRAIN_RATIO + VAL_RATIO)), n)}


# ------------------------------------------------------------------ 1-4  daily interval series
def daily_btc(csv):
    df = pd.read_csv(csv, usecols=["system_time", "midpoint", "spread"])
    t = pd.to_datetime(df.system_time, utc=True, format="ISO8601").dt.tz_localize(None)
    g = pd.DataFrame({"day": t.dt.floor("D"), "lo": df.midpoint - df.spread / 2, "hi": df.midpoint + df.spread / 2,
                      "close": df.midpoint}).groupby("day")
    return pd.DataFrame({"btc_lo": g.lo.min(), "btc_hi": g.hi.max(), "btc_close": g.close.last()})


def daily_macro(raw, days):
    out = {}
    for v in MACRO:
        m = pd.read_csv(raw / f"{v}.csv", parse_dates=["date"]).set_index("date").sort_index()
        m = m.reindex(m.index.union(days)).ffill().reindex(days)        # weekends / holidays: last trading day
        out[f"{v}_lo"], out[f"{v}_hi"] = m.Low, m.High
    return pd.DataFrame(out, index=days)


def stitch_trends(raw):
    """Chain the 200-day windows on their overlaps; window 0 is the anchor, later windows are rescaled to it."""
    w = pd.read_csv(raw / "gtrends_daily_windows.csv", parse_dates=["date"])
    w = w[~w.isPartial.astype(str).str.lower().eq("true")]
    series = None
    for i, g in w.groupby("window"):
        s = g.set_index("date").Bitcoin.astype(float)
        if series is None:
            series = s
            continue
        common = series.index.intersection(s.index)
        common = common[(series[common] > 0) & (s[common] > 0)]
        factor = series[common].sum() / s[common].sum()
        series = pd.concat([series, s[~s.index.isin(series.index)] * factor]).sort_index()
    return series


def daily_trends(raw, days):
    s = stitch_trends(raw).shift(1, freq="D")                           # published a day late
    s = s.reindex(s.index.union(days)).ffill()
    lo = s.rolling(GT_WINDOW, min_periods=1).min().reindex(days)
    hi = s.rolling(GT_WINDOW, min_periods=1).max().reindex(days)
    return pd.DataFrame({"gt_lo": lo, "gt_hi": hi}, index=days)


def vader_scores(tweets_csv, out_csv):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    df = pd.read_csv(tweets_csv, usecols=["id", "created_at", "text", "sentiment_score", "risk_score"])
    df["vader"] = [an.polarity_scores(_URL.sub("", str(t)).strip())["compound"] for t in df.text]
    df.drop(columns="text").to_csv(out_csv, index=False)
    return df


def binned_quantile(scores, q):
    """Quantile of integer 1-5 scores, each score spread uniformly over [k - 0.5, k + 0.5] (continuity
    correction): moves with the mix of scores instead of snapping to an integer."""
    p = np.bincount(np.asarray(scores, int), minlength=6)[1:6] / len(scores)
    cdf = np.concatenate([[0.0], np.cumsum(p)])
    k = int(np.searchsorted(cdf, q, side="left"))                      # 1-based bin holding the quantile
    k = min(max(k, 1), 5)
    return k - 0.5 + (q - cdf[k - 1]) / p[k - 1] if p[k - 1] > 0 else float(k)


def daily_sentiment(scores_csv, days):
    """Daily [q25, q75]. Two deviations from the paper, needed on tweets (see EXPERIMENTS.md, E13):
    VADER ignores tweets scored exactly 0 (25 % of tweets: no sentiment words; with them q25 = 0 on 83 % of
    days); DeepSeek's integer scores use the continuity-corrected quantile (plain quartiles are [3, 4] on 95 %
    of days)."""
    df = pd.read_csv(scores_csv)
    day = pd.to_datetime(df.created_at, utc=True, format="ISO8601").dt.tz_localize(None).dt.floor("D")
    df = df.assign(day=day)
    out = pd.DataFrame(index=days)
    nz = df[df.vader != 0].groupby("day").vader
    out["vader_lo"], out["vader_hi"] = nz.quantile(0.25).reindex(days), nz.quantile(0.75).reindex(days)
    g = df.groupby("day")
    for name, col in (("deepseek", "sentiment_score"), ("risk", "risk_score")):
        out[f"{name}_lo"] = g[col].apply(lambda s: binned_quantile(s, 0.25)).reindex(days)
        out[f"{name}_hi"] = g[col].apply(lambda s: binned_quantile(s, 0.75)).reindex(days)
    out["n_tweets"] = g.size().reindex(days).fillna(0).astype(int)
    out["n_tweets_vader_nonzero"] = nz.size().reindex(days).fillna(0).astype(int)
    return out


# ------------------------------------------------------------------ 5  NA-MEMD
def _maxima(P):
    """Interior local maxima of every column of P (N x K) -> list of index arrays."""
    up = (P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])
    return [np.flatnonzero(up[:, k]) + 1 for k in range(P.shape[1])]


def _envelope_mean(h, dirs, t):
    P = h @ dirs.T
    n, acc, used = len(h), np.zeros_like(h), 0
    for idx in _maxima(P):
        if len(idx) < 2:
            continue
        left, right = idx[:2][::-1], idx[-2:][::-1]                     # mirror two maxima at each end
        tt = np.concatenate([-t[left], t[idx], 2 * (n - 1) - t[right]])
        yy = np.concatenate([h[left], h[idx], h[right]])
        acc += CubicSpline(tt, yy, axis=0)(t)
        used += 1
    return acc / used if used else None


def memd(X, rng, n_dirs=N_DIRS, n_sift=N_SIFT, max_imfs=MAX_IMFS):
    """Multivariate EMD (Rehman & Mandic 2010), fixed number of sifting steps. X: N x C -> (IMFs K x N x C, residue)."""
    n, c = X.shape
    dirs = rng.normal(size=(n_dirs, c))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    t = np.arange(n, dtype=float)
    r, imfs = X.copy(), []
    while len(imfs) < max_imfs:
        if np.median([len(i) for i in _maxima(r @ dirs.T)]) < 2:
            break
        h = r.copy()
        for _ in range(n_sift):
            m = _envelope_mean(h, dirs, t)
            if m is None:
                break
            h = h - m
        imfs.append(h)
        r = r - h
    return np.array(imfs), r


def na_memd(X, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=NOISE_STD, size=(len(X), NOISE_CHANNELS))
    imfs, res = memd(np.hstack([X, noise]), rng)
    return imfs[:, :, : X.shape[1]], res[:, : X.shape[1]]


# ------------------------------------------------------------------ 6  sample entropy grouping
def sample_entropy(x, m=SE_M, r=SE_R):
    x = np.asarray(x, float)
    tol = r * x.std()
    if tol == 0 or len(x) <= m + 1:
        return 0.0

    def pairs(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: len(x) - m]
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=2)
        return ((d <= tol).sum() - len(emb)) / 2

    b, a = pairs(m), pairs(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else float(np.log(len(x) ** 2))


def group_bands(orig, imfs, res):
    """orig N x 2, imfs K x N x 2, res N x 2 of ONE variable -> {'low'|'medium'|'high': N x 2}."""
    se = lambda y: np.mean([sample_entropy(y[:, j]) for j in range(y.shape[1])])
    se_orig, se_imf = se(orig), [se(f) for f in imfs]
    low = [k for k, s in enumerate(se_imf) if s < se_orig]
    rest = [k for k in range(len(imfs)) if k not in low]
    high, medium = [], []
    if rest:
        thr = se(imfs[rest].sum(axis=0))
        high = [k for k in rest if se_imf[k] > thr]
        medium = [k for k in rest if k not in high]
    if not high and len(imfs):
        # Adaptation (not in the paper): on noisy daily series (our tweet sentiment, some BTC days) every IMF
        # is more regular than the series itself, so the rule leaves the high band empty. The first MEMD IMF
        # is by construction the fastest oscillation, so it becomes the high band.
        low, medium, high = [k for k in low if k != 0], [k for k in medium if k != 0], [0]
    zero = np.zeros_like(orig)
    pick = lambda ks: imfs[ks].sum(axis=0) if ks else zero
    return {"low": pick(low) + res, "medium": pick(medium), "high": pick(high)}, {"low": low, "medium": medium, "high": high}


# ------------------------------------------------------------------ causal decomposition -> features
def _day_features(args):
    """Decompose the series up to day `t` (inclusive) and return that day's features for both scorers."""
    t, panel = args
    out = {"day_index": t}
    for scorer in SCORERS:
        cols = [f"{v}_{b}" for v in VARIABLES for b in ("lo", "hi")]
        Z = panel[scorer][: t + 1]
        mu, sd = Z.mean(axis=0), Z.std(axis=0)
        sd[sd == 0] = 1.0
        imfs, res = na_memd((Z - mu) / sd, seed=t)
        bands, groups = {}, {}
        for i, v in enumerate(VARIABLES):
            j = [2 * i, 2 * i + 1]
            b, g = group_bands(Z[:, j], imfs[:, :, j] * sd[j], res[:, j] * sd[j] + mu[j])
            bands[v], groups[v] = b, g
        c = lambda v, band, k=-1: bands[v][band][k].mean()            # centre of a band at day k
        w = lambda v, band, k=-1: bands[v][band][k][1] - bands[v][band][k][0]
        price = Z[-1, :2].mean()
        f = {"btc_trend_gap": price / c("btc", "low") - 1, "btc_cycle": c("btc", "medium") / price,
             "btc_noise": c("btc", "high") / price, "btc_noise_width": w("btc", "high") / price}
        for v in MACRO:
            f[f"{v}_trend"] = c(v, "low") / c(v, "low", -6) - 1
        gt_level = max(abs(c("gt", "low")), 1e-9)
        f["gt_cycle"], f["gt_cycle_width"] = c("gt", "medium") / gt_level, w("gt", "medium") / gt_level
        f["sent_noise"], f["sent_noise_width"] = c("sent", "high"), w("sent", "high")
        out.update({f"{scorer}__{k}": float(x) for k, x in f.items()})
        out[f"{scorer}__n_imfs"] = int(len(imfs))
        out[f"{scorer}__groups"] = json.dumps({v: groups[v] for v in VARIABLES})
    return out


def build(data, btc_csv, workers=6, days_subset=None):
    data = Path(data)
    raw = data / "raw"
    btc = daily_btc(btc_csv).loc[START:END]
    days = btc.index
    sent = daily_sentiment(data / "tweet_scores_vader.csv", days)
    df = pd.concat([btc, daily_macro(raw, days), daily_trends(raw, days), sent], axis=1)
    for name, (a, b) in EVENTS.items():
        df[name] = ((df.index >= a) & (df.index <= b)).astype(int)
    if df.drop(columns=list(EVENTS)).isna().any().any():
        raise ValueError("missing values: " + str(df.isna().sum()[df.isna().sum() > 0].to_dict()))
    panel = {}
    for scorer in SCORERS:
        cols = [f"{v}_{b}" for v in VARIABLES[:-1] for b in ("lo", "hi")] + [f"{scorer}_lo", f"{scorer}_hi"]
        panel[scorer] = df[cols].to_numpy(float)
    todo = list(range(MIN_DAYS - 1, len(df))) if days_subset is None else list(days_subset)
    with Pool(workers) as pool:
        rows = pool.map(_day_features, [(t, panel) for t in todo], chunksize=4)
    feats = pd.DataFrame(rows).set_index("day_index").sort_index()
    feats.index = df.index[feats.index]
    return df, feats


def scale(df_feats, n_days):
    """Train mean / train 99th percentile of |x - mean| -> [-1, 1]; days without a decomposition -> 0."""
    tr = split_days(n_days)["train"]
    out, info = pd.DataFrame(index=df_feats.index), {}
    for c in df_feats.columns:
        x = df_feats[c].astype(float)
        train = x.iloc[tr[0]:tr[1]].dropna()
        mu = float(train.mean())
        s = float(np.quantile(np.abs(train - mu), 0.99)) or 1.0
        out[c] = ((x - mu) / s).clip(-1, 1).fillna(0.0)
        info[c] = {"mean": mu, "scale": s}
    return out, info


# ------------------------------------------------------------------ pre-check
def precheck(data, n_null=500, seed=0):
    """Spearman IC of each lagged feature with the next day's BTC return and absolute return, per split,
    with a circular-shift null (shifts of at least 30 days)."""
    data = Path(data)
    df = pd.read_csv(data / "daily_intervals.csv", index_col=0, parse_dates=True)
    f = pd.read_csv(data / "daily_features_scaled.csv", index_col=0, parse_dates=True)
    ret = df.btc_close.pct_change()                          # return over day D (close D-1 -> close D)
    target = {"return": ret, "abs_return": ret.abs()}
    x_all = f.shift(1)                                       # day D uses features of day D-1
    rng = np.random.default_rng(seed)
    rows = []
    for split, (a, b) in split_days(len(df)).items():
        if split == "test":                                  # the test split is not looked at before the runs
            continue
        for col in f.columns:
            for tname, y in target.items():
                xs, ys = x_all[col].iloc[a:b], y.iloc[a:b]
                ok = xs.notna() & ys.notna() & (xs != 0)
                xs, ys = xs[ok].to_numpy(), ys[ok].to_numpy()
                if len(xs) < 30 or np.std(xs) == 0:
                    continue
                ic = spearmanr(xs, ys).statistic
                null = [spearmanr(np.roll(xs, k), ys).statistic for k in rng.integers(30, len(xs) - 30, n_null)]
                p = (1 + np.sum(np.abs(null) >= abs(ic))) / (1 + n_null)
                rows.append({"split": split, "feature": col, "target": tname, "n": len(xs), "ic": ic, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(data / "precheck_ic.csv", index=False)
    return out


# ------------------------------------------------------------------ onto the 1-minute bars
def prepare(data, prepared_dir, out_dir):
    """E5's prepared market arrays + the E13 features, lagged one UTC day (bars on day D see day D-1's
    features, which use data up to the end of D-1), as `e13_{scorer}__{feature}`; plus `gate_open_{scorer}`:
    0 while day D-1's raw sentiment interval is wider than its train-days median (state H)."""
    data, prepared_dir, out_dir = Path(data), Path(prepared_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    iv = pd.read_csv(data / "daily_intervals.csv", index_col=0, parse_dates=True)
    f = pd.read_csv(data / "daily_features_scaled.csv", index_col=0, parse_dates=True).reindex(iv.index).fillna(0.0)
    tr = split_days(len(iv))["train"]
    lagged = f.shift(1).fillna(0.0)
    gates = {}
    for s in SCORERS:
        width = iv[f"{s}_hi"] - iv[f"{s}_lo"]
        med = float(width.iloc[tr[0]:tr[1]].median())
        gates[s] = ((width <= med).astype(float).shift(1).fillna(1.0), med)
    info = {"lag_days": 1, "gate_median_width": {s: gates[s][1] for s in SCORERS}, "splits": {}}
    for split in ("train", "val", "test"):
        m = dict(np.load(prepared_dir / f"market_{split}.npz"))
        t = m["time_ns"]                                     # despite the name, prepare.py stores microseconds
        unit = "ns" if t.max() > 1e17 else "us"
        day = pd.to_datetime(t, unit=unit, utc=True).tz_localize(None).floor("D")
        day = pd.DatetimeIndex(day).as_unit(iv.index.unit)
        pos = iv.index.get_indexer(day)
        if (pos < 0).any():
            raise ValueError(f"{split}: {(pos < 0).sum()} bars outside the daily range")
        for c in f.columns:
            scorer, feat = c.split("__")
            m[f"e13_{scorer}__{feat}"] = lagged[c].to_numpy(np.float32)[pos]
        for s in SCORERS:
            m[f"gate_open_{s}"] = gates[s][0].to_numpy(np.float32)[pos]
        np.savez(out_dir / f"market_{split}.npz", **m)
        info["splits"][split] = {"bars": len(pos), "first_day": str(day[0].date()), "last_day": str(day[-1].date()),
                                 **{f"gate_open_{s}": float(m[f"gate_open_{s}"].mean()) for s in SCORERS}}
    meta = json.loads((prepared_dir / "meta.json").read_text())
    meta["e13"] = info
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rltrade.paper_daily")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vader"); v.add_argument("--data", required=True); v.add_argument("--tweets", required=True)
    b = sub.add_parser("build"); b.add_argument("--data", required=True); b.add_argument("--btc", required=True)
    b.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("precheck"); p.add_argument("--data", required=True)
    r = sub.add_parser("prepare"); r.add_argument("--data", required=True); r.add_argument("--prepared", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(a.data, a.prepared, a.out), indent=1))
    if a.cmd == "vader":
        df = vader_scores(a.tweets, Path(a.data) / "tweet_scores_vader.csv")
        print(f"{len(df):,} tweets scored; VADER exactly 0: {(df.vader == 0).mean():.1%}; "
              f"Spearman with DeepSeek: {spearmanr(df.vader, df.sentiment_score).statistic:.3f}")
    elif a.cmd == "build":
        df, feats = build(a.data, a.btc, a.workers)
        df.to_csv(Path(a.data) / "daily_intervals.csv")
        feats.to_csv(Path(a.data) / "daily_features_raw.csv")
        num = feats[[c for c in feats.columns if "__" in c and not c.endswith(("n_imfs", "groups"))]]
        scaled, info = scale(num.reindex(df.index), len(df))
        scaled.to_csv(Path(a.data) / "daily_features_scaled.csv")
        meta = {"days": len(df), "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
                "splits": {k: [str(df.index[i].date()), str(df.index[j - 1].date())] for k, (i, j) in split_days(len(df)).items()},
                "min_days": MIN_DAYS, "memd": {"dirs": N_DIRS, "sift": N_SIFT, "max_imfs": MAX_IMFS,
                "noise_channels": NOISE_CHANNELS, "noise_std": NOISE_STD}, "sample_entropy": {"m": SE_M, "r": SE_R},
                "scaling": info}
        (Path(a.data) / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{len(df)} days, {len(feats)} decomposed, features -> {Path(a.data).resolve()}")
    elif a.cmd == "precheck":
        out = precheck(a.data)
        print(out.sort_values("p").head(30).to_string(index=False))


if __name__ == "__main__":
    main()
