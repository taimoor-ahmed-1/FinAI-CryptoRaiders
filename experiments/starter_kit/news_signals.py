"""
Experiment 11 - the organisers' DeepSeek news scores mapped onto every second of the contest file.

    python news_signals.py BTC_1sec.csv news_train.csv PREPARED_DIR     -> BTC_1sec_signals.npy (+ .json)

The kit (and the organisers' merged csv) held the latest article's score until the next article arrived,
so a single article could set the signal for hours, and a burst of articles counted no more than one.
Here each second's value is a decayed, confidence-weighted average over the relevant articles already
published, which fades back to "no news" when nothing new arrives:

    available_i  max(date_time, the time stamped in the article's Yahoo URL) + LATENCY   (as in E9, causal)
    w_i(t)       confidence_i * 2 ** (-(t - available_i) / HALF_LIFE)        for t >= available_i, else 0
    s(t)         sum_i w_i(t) * (score_i - centre) / (PRIOR + sum_i w_i(t))

    centre       the confidence-weighted mean score of the relevant articles available before the end of
                 the train split, so "no news" reads as a typical day, not as bearish (most articles score 4)
    PRIOR        weight of a neutral pseudo-article: one fresh article moves the signal ~46 % of the way
                 to its score, several agreeing articles move it further
    relevance    title or the first 600 characters of the text mention bitcoin / crypto (232 of 279)

Output columns: sentiment, risk; each divided by the 99th percentile of |s| over the train split and clipped
to [-1, 1], the range of the other state inputs (0 stays "no news"; raw values are small, ~0.1 on average).
The settings are fixed in advance (not tuned on validation or test).
"""
import json
import re
import sys

import numpy as np
import pandas as pd

HALF_LIFE = 3600.0  # seconds; the median gap between articles is 15 min, the 90th percentile 65 min
LATENCY = 60.0  # seconds between publication and the score being usable (scraping + DeepSeek call)
PRIOR = 1.0
SCORES = ("sentiment_score", "risk_score")
RELEVANT = re.compile(r"bitcoin|\bbtc\b|crypto|blockchain|coinbase|ether|\beth\b|miner|mining|digital asset"
                      r"|digital currenc|altcoin|defi|stablecoin|satoshi|grayscale|microstrategy|dogecoin|binance")
_YAHOO_TIME = re.compile(r"-(\d{2})(\d{2})(\d{2})\d{3}\.html")


def news_availability(news):
    """Time from which each article may be used: max(date_time, Yahoo URL time) - as in rltrade/contest.py (E9)."""
    t = pd.to_datetime(news["date_time"], utc=True, format="ISO8601")
    avail = t.copy()
    for i, (ti, url) in enumerate(zip(t, news["url"].fillna(""))):
        m = _YAHOO_TIME.search(url) if "finance.yahoo.com" in url else None
        if not m:
            continue
        h, mi, s = map(int, m.groups())
        u = ti.normalize() + pd.Timedelta(hours=h, minutes=mi, seconds=s)
        if u < ti - pd.Timedelta(hours=12):  # URL clock is past midnight: next calendar day
            u += pd.Timedelta(days=1)
        avail.iloc[i] = max(ti, u)
    return avail


def relevant(news):
    text = (news["title"].fillna("") + " " + news["article_text"].fillna("").str[:600]).str.lower()
    return text.str.contains(RELEVANT).to_numpy()


def decayed_signals(row_times, news, train_end, half_life=HALF_LIFE, latency=LATENCY, prior=PRIOR):
    """(n_rows, 2) float32: decayed, confidence-weighted, centred scores; row i uses only articles available
    at or before row_times[i]."""
    news = news[relevant(news)].copy()
    news["available"] = news_availability(news) + pd.Timedelta(seconds=latency)
    t = (row_times - row_times.iloc[0]).dt.total_seconds().to_numpy()
    a = (news["available"] - row_times.iloc[0]).dt.total_seconds().to_numpy()
    conf = {"sentiment_score": news["confidence_score_sentiment"].to_numpy(float),
            "risk_score": news["confidence_score_risk"].to_numpy(float)}
    in_train = (news["available"] < train_end).to_numpy()
    t_train = (row_times < train_end).to_numpy()
    out, centres, scales = [], {}, {}
    for col in SCORES:
        x, c = news[col].to_numpy(float), conf[col]
        centre = float(np.average(x[in_train], weights=c[in_train]))
        num, den = np.zeros(len(t)), np.zeros(len(t))
        for ai, xi, ci in zip(a, x, c):
            k = np.searchsorted(t, ai)  # first row at or after the article becomes available
            w = ci * np.exp2(-(t[k:] - ai) / half_life)
            num[k:] += w * (xi - centre)
            den[k:] += w
        s = num / (prior + den)
        scale = float(np.quantile(np.abs(s[t_train]), 0.99))
        out.append(np.clip(s / scale, -1, 1))
        centres[col], scales[col] = centre, scale
    meta = {"half_life_s": half_life, "latency_s": latency, "prior": prior, "centres": centres, "scales": scales,
            "articles": int(len(news)), "articles_in_train": int(in_train.sum())}
    return np.stack(out, axis=1).astype(np.float32), meta


def build(sec_csv, news_csv, data_dir):
    from data_config import ConfigData
    cfg = ConfigData(data_dir)
    times = pd.to_datetime(pd.read_csv(sec_csv, usecols=["system_time"])["system_time"], utc=True, format="ISO8601")
    n_rows = np.load(cfg.price_ary_path, mmap_mode="r").shape[0]
    assert len(times) == n_rows, f"{sec_csv} has {len(times)} rows, the prepared arrays {n_rows}"
    train_end = times.iloc[cfg.split_rows("train")[1]]
    sig, meta = decayed_signals(times, pd.read_csv(news_csv), train_end)
    for split in ("train", "val", "test"):
        r0, r1 = cfg.split_rows(split)
        meta[f"{split}_nonzero_share"] = [round(float((sig[r0:r1, j] != 0).mean()), 3) for j in range(2)]
        meta[f"{split}_mean_abs"] = [round(float(np.abs(sig[r0:r1, j]).mean()), 4) for j in range(2)]
    np.save(cfg.signals_ary_path, sig)
    json.dump(meta, open(cfg.signals_ary_path.replace(".npy", ".json"), "w"), indent=1)
    print(f"| news signals: {sig.shape} -> {cfg.signals_ary_path}\n{json.dumps(meta, indent=1)}")
    return sig, meta


if __name__ == "__main__":
    build(*sys.argv[1:4])
