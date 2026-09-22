"""The LLM signals seen by the agent must come only from COMPLETED days.

The dataset repeats each day's tweet aggregate on every bar of that same day, so
a bar at 00:05 would see tweets from later that day. `prepare.lag_daily_signals`
must hand every bar of day D the aggregate of day D-1.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade.prepare import lag_daily_signals   # noqa: E402


def test_bars_see_previous_day_only():
    t = pd.date_range("2024-01-01", periods=4 * 1440, freq="1min", tz="UTC")
    day_vals = {0: 3.0, 1: 4.0, 2: 2.0, 3: 5.0}
    df = pd.DataFrame({"system_time": t.astype(str),
                       "sentiment_score": [day_vals[i // 1440] for i in range(len(t))]})
    lag = lag_daily_signals(df, ["sentiment_score"], train_end=2 * 1440)["sentiment_score"]
    for d in (1, 2, 3):
        bars = lag[d * 1440:(d + 1) * 1440]
        assert np.all(bars == day_vals[d - 1]), f"day {d} does not see day {d - 1}"
    assert np.allclose(lag[:1440], np.mean([3.0, 4.0]))      # first day: train-period mean


def test_calendar_gap_uses_last_earlier_day():
    t = pd.DatetimeIndex(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-03 09:00"], tz="UTC")
    df = pd.DataFrame({"system_time": t.astype(str), "risk_score": [1.0, 1.0, 5.0]})
    lag = lag_daily_signals(df, ["risk_score"], train_end=3)["risk_score"]
    assert lag[2] == 1.0


def test_real_data_is_daily_constant_and_lagged():
    p = os.environ.get("BTC_CSV", "")
    if not (p and Path(p).exists()):
        print("  (BTC_CSV not set - real-data check skipped)")
        return
    df = pd.read_csv(p, nrows=5 * 1440, usecols=["system_time", "sentiment_score", "risk_score"])
    day = pd.to_datetime(df.system_time, utc=True).dt.floor("D")
    assert (df.groupby(day)[["sentiment_score", "risk_score"]].nunique() == 1).all().all(), \
        "signals vary within a day - the lag logic assumes a daily aggregate"
    lag = lag_daily_signals(df, ["sentiment_score"], train_end=len(df))["sentiment_score"]
    raw = df.sentiment_score.to_numpy(np.float32)
    assert np.array_equal(lag[1440:2 * 1440], raw[:1440])   # day 2 sees day 1
    print("  real data: daily-constant signals, lag verified")


if __name__ == "__main__":
    test_bars_see_previous_day_only();       print("lag ................ OK")
    test_calendar_gap_uses_last_earlier_day(); print("calendar gap ....... OK")
    test_real_data_is_daily_constant_and_lagged(); print("real data .......... OK")
