"""Contest-week dataset (rltrade/contest.py): bars, news availability and causal signals.

A bar must only see articles available before the bar ends, and an article whose Yahoo URL
time is later than its `date_time` must wait until that later time.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import contest                     # noqa: E402
from rltrade.prepare import prepare             # noqa: E402


def _sec(n=180, start="2021-04-07 12:00:00"):
    t = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    d = {"system_time": t.strftime("%Y-%m-%d %H:%M:%S+00:00"), "midpoint": 100.0 + np.arange(n) % 60,
         "buys": np.ones(n), "sells": 2 * np.ones(n)}
    for side in ("bids", "asks"):
        for i in contest.LEVELS:
            d[f"{side}_notional_{i}"] = np.arange(n, dtype=float)
    return pd.DataFrame(d)


def _news(rows):
    return pd.DataFrame(rows, columns=["date_time", "url", "sentiment_score", "risk_score"])


def test_bars():
    b = contest.build_bars(_sec())
    assert len(b) == 3 and b.system_time.iloc[0].startswith("2021-04-07T12:00:00")
    assert np.allclose(b.midpoint, 129.5) and np.allclose(b.spread, 59.0)        # (100 + 159) / 2, 159 - 100
    assert np.allclose(b.buys, 60) and np.allclose(b.sells, 120)
    assert np.allclose(b.bids_notional_3, 15 * np.array([59, 119, 179]))        # 15 levels at the last second
    assert np.allclose(b.bids_distance_3, (100 - 129.5) / 129.5)


def test_availability_uses_later_url_time():
    n = _news([("2021-04-07 12:00:00+00:00", "https://finance.yahoo.com/news/a-200000331.html", 3, 3),   # URL 8 h later
               ("2021-04-07 18:18:00+00:00", "https://finance.yahoo.com/news/b-120000111.html", 3, 3),   # URL earlier
               ("2021-04-08 19:52:00+00:00", "https://finance.yahoo.com/news/c-023000222.html", 3, 3),   # next day
               ("2021-04-07 13:05:07+00:00", "https://www.coindesk.com/x", 3, 3)])
    av = contest.news_availability(n)
    assert av.iloc[0] == pd.Timestamp("2021-04-07 20:00:00", tz="UTC")
    assert av.iloc[1] == pd.Timestamp("2021-04-07 18:18:00", tz="UTC")
    assert av.iloc[2] == pd.Timestamp("2021-04-09 02:30:00", tz="UTC")
    assert av.iloc[3] == pd.Timestamp("2021-04-07 13:05:07", tz="UTC")


def test_signals_are_causal():
    bars = pd.Series(pd.date_range("2021-04-07 11:58", periods=6, freq="1min", tz="UTC")
                     .strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    n = _news([("2021-04-07 12:00:30+00:00", "", 4, 2),          # inside bar 12:00 -> seen from bar 12:00 on
               ("2021-04-07 12:02:00+00:00", "", 2, 4),          # exactly at the end of bar 12:01 -> from bar 12:02
               ("2021-04-07 12:02:00+00:00", "", 3, 5)])         # same time: averaged
    s = contest.news_signals(bars, n)
    assert s.sentiment_score.iloc[:2].isna().all()                # 11:58, 11:59: nothing available yet
    assert s.sentiment_score.iloc[2] == 4 and s.sentiment_score.iloc[3] == 4
    assert s.sentiment_score.iloc[4] == 2.5 and s.risk_score.iloc[4] == 4.5


def test_prepare_keeps_causal_signals():
    n = 3000
    t = pd.date_range("2021-04-07", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"system_time": t.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                       "midpoint": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))), "spread": 0.1,
                       "buys": 1.0, "sells": 1.0, "bids_distance_3": -1e-4, "asks_distance_3": 1e-4,
                       "bids_notional_3": 5.0, "asks_notional_3": 5.0,
                       "sentiment_score": rng.integers(1, 6, n).astype(float), "risk_score": 3.0})
    with tempfile.TemporaryDirectory() as d:
        df.to_csv(Path(d) / "x.csv", index=False)
        meta = prepare(Path(d) / "x.csv", Path(d) / "p", signal_lag="none", verbose=False)
        a, b = meta["splits"]["val"]["rows"]
        with np.load(Path(d) / "p" / "market_val.npz") as m:
            assert np.array_equal(m["sentiment_score"], df.sentiment_score.to_numpy(np.float32)[a:b])
            assert np.array_equal(m["sameday__sentiment_score"], m["sentiment_score"])
        assert meta["signal_lag"] == "none"


def test_group_specs():
    hp = {"lr": 1e-4}
    ids = [s.run_id for g in contest.GROUPS for s in contest.group_specs(g, hp, {"reward_scale": 100.0})]
    assert len(ids) == len(set(ids)) == 4 * 10 + 2 * 10 + 4
    low = contest.group_specs("D_lowcost", hp, {})[0]
    assert low.env.cost_rate == 0.0002 and low.tag == "lowcost" and low.env.state == "D"
    real = contest.group_specs("B", hp, {})[0]
    assert abs(real.env.cost_rate - 0.0012) < 1e-12 and real.env.episode_bars == 720 and real.val_chunks == 1


if __name__ == "__main__":
    test_bars();                          print("bars ............... OK")
    test_availability_uses_later_url_time(); print("availability ....... OK")
    test_signals_are_causal();            print("causal signals ..... OK")
    test_prepare_keeps_causal_signals();  print("prepare no-lag ..... OK")
    test_group_specs();                   print("group specs ........ OK")
