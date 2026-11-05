"""
Experiment 12 - news signals built the way Tian et al. (2026, Pacific-Basin Finance Journal 103352) build
theirs, mapped onto every second of the contest file.

    python vader_signals.py BTC_1sec.csv news_train.csv PREPARED_DIR     -> BTC_1sec_signals_vader.npy (+ .json)

The paper scores CoinDesk headlines plus their summaries with VADER (lexicon and rules, compound score in
[-1, 1]) and turns a week of article scores into an interval: the 25th and 75th percentiles, which ignore
single outlier articles. They have ~28 articles per week (17,383 over 628 weeks); here the interval at second t
is taken over the last WINDOW relevant articles already available, so it holds as many articles as theirs:

    available_i  max(date_time, Yahoo URL time) + LATENCY        (as in E9 / E11, causal)
    score_i      VADER compound of the title + the article's first paragraph (<= 600 characters), the
                 nearest thing to the paper's headline + summary
    interval(t)  [q25, q75] of score_i over the last WINDOW articles with available_i <= t (>= MIN_ARTICLES)

Output columns: interval centre (q25 + q75) / 2 = sentiment, and width q75 - q25 = how much the news
disagrees, the stand-in for risk (the paper has no risk score; its risk handling is the interval). Each is
centred on its train-split mean and divided by the train 99th percentile of |value - mean|, clipped to
[-1, 1]; rows before MIN_ARTICLES articles are available are 0. Not used from the paper: the NA-MEMD
decomposition (it decomposes the whole series at once - look-ahead) and Google Trends.
The settings are fixed in advance (not tuned on validation or test).
"""
import json
import sys

import numpy as np
import pandas as pd

from news_signals import LATENCY, news_availability, relevant

WINDOW = 28  # articles per interval: the paper's 17,383 articles / 628 weeks
MIN_ARTICLES = 5
COLUMNS = ("vader_centre", "vader_width")


def article_text(news):
    """Title + first paragraph, at most 600 characters."""
    first = news["article_text"].fillna("").str.split("\n").str[0].str[:600]
    return (news["title"].fillna("") + ". " + first).tolist()


def vader_scores(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


def interval_signals(row_times, news, train_end, window=WINDOW, min_articles=MIN_ARTICLES, latency=LATENCY):
    """(n_rows, 2) float32: centre and width of the rolling [q25, q75] VADER interval; row i uses only
    articles available at or before row_times[i]."""
    news = news[relevant(news)].copy()
    news["available"] = news_availability(news) + pd.Timedelta(seconds=latency)
    news["vader"] = vader_scores(article_text(news))
    news = news.sort_values("available", kind="stable")
    t = (row_times - row_times.iloc[0]).dt.total_seconds().to_numpy()
    a = (news["available"] - row_times.iloc[0]).dt.total_seconds().to_numpy()
    x = news["vader"].to_numpy()
    # interval after the k-th article becomes available (NaN while fewer than min_articles)
    lo, hi = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for k in range(len(x)):
        w = x[max(0, k - window + 1):k + 1]
        if len(w) >= min_articles:
            lo[k], hi[k] = np.quantile(w, 0.25), np.quantile(w, 0.75)
    k_row = np.searchsorted(a, t, side="right") - 1  # last article available at or before each row
    ok = k_row >= 0
    raw = np.full((len(t), 2), np.nan)
    raw[ok, 0] = (lo[k_row[ok]] + hi[k_row[ok]]) / 2
    raw[ok, 1] = hi[k_row[ok]] - lo[k_row[ok]]
    t_train = (row_times < train_end).to_numpy()
    out, centres, scales = [], {}, {}
    for j, col in enumerate(COLUMNS):
        v = raw[:, j]
        ref = v[t_train & ~np.isnan(v)]
        centre = float(ref.mean())
        scale = float(np.quantile(np.abs(ref - centre), 0.99))
        out.append(np.nan_to_num(np.clip((v - centre) / scale, -1, 1), nan=0.0))
        centres[col], scales[col] = centre, scale
    spans = pd.Series(a).diff(window - 1).dropna() / 3600  # hours covered by a full window
    meta = {"window_articles": window, "min_articles": min_articles, "latency_s": latency,
            "centres": centres, "scales": scales, "articles": int(len(news)),
            "articles_in_train": int((news["available"] < train_end).sum()),
            "window_span_hours_median": round(float(spans.median()), 2),
            "vader_mean": round(float(x.mean()), 3), "vader_share_zero": round(float((x == 0).mean()), 3),
            "vader_vs_deepseek_sentiment_spearman": round(float(pd.Series(x).corr(
                news["sentiment_score"].reset_index(drop=True), method="spearman")), 3)}
    return np.stack(out, axis=1).astype(np.float32), meta


def build(sec_csv, news_csv, data_dir):
    from data_config import ConfigData
    cfg = ConfigData(data_dir)
    path = cfg.signals_ary_path.replace(".npy", "_vader.npy")
    times = pd.to_datetime(pd.read_csv(sec_csv, usecols=["system_time"])["system_time"], utc=True, format="ISO8601")
    n_rows = np.load(cfg.price_ary_path, mmap_mode="r").shape[0]
    assert len(times) == n_rows, f"{sec_csv} has {len(times)} rows, the prepared arrays {n_rows}"
    train_end = times.iloc[cfg.split_rows("train")[1]]
    sig, meta = interval_signals(times, pd.read_csv(news_csv), train_end)
    for split in ("train", "val", "test"):
        r0, r1 = cfg.split_rows(split)
        meta[f"{split}_nonzero_share"] = [round(float((sig[r0:r1, j] != 0).mean()), 3) for j in range(2)]
        meta[f"{split}_mean_abs"] = [round(float(np.abs(sig[r0:r1, j]).mean()), 4) for j in range(2)]
    np.save(path, sig)
    json.dump(meta, open(path.replace(".npy", ".json"), "w"), indent=1)
    print(f"| VADER interval signals: {sig.shape} -> {path}\n{json.dumps(meta, indent=1)}")
    return sig, meta


if __name__ == "__main__":
    build(*sys.argv[1:4])
