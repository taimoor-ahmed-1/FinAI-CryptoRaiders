"""The notebook's original helper operators, kept verbatim for equivalence tests.
Do not use in production - they are exact but very slow."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

WINDOW = 10
PERIOD = 10

def ref(s, n=1):
    return pd.Series(s).shift(n).values

def ts_sum(df, window=WINDOW):
    return df.rolling(window).sum()

def sma(df, window=WINDOW):
    return df.rolling(window).mean()

def ema(df, window, *, adjust=True, min_periods=1):
    return df.ewm(ignore_na=False, span=window, min_periods=min_periods, adjust=adjust).mean()

def stddev(df, window=WINDOW):
    return df.rolling(window).std()

def correlation(x, y, window=WINDOW):
    return x.rolling(window).corr(y)

def covariance(x, y, window=WINDOW):
    return x.rolling(window).cov(y)

def rolling_rank(na):
    return rankdata(na)[-1]

def ts_rank(df, window=WINDOW):
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    return np.prod(na)

def product(df, window=WINDOW):
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=WINDOW):
    return df.rolling(window).min()

def ts_max(df, window=WINDOW):
    return df.rolling(window).max()

def df_delta(df, period=1):
    return df.diff(period)

def delay(df, period=1):
    return df.shift(period)

def rank(df, window_size=WINDOW):
    return df.rolling(window=window_size).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

def scale(df, window_size=WINDOW, k=1):
    scaled = df.mul(k)
    rolling_sums = np.abs(df).rolling(window=window_size, min_periods=1).sum()
    normalized = scaled.div(rolling_sums)
    return normalized

def ts_argmax(df, window=WINDOW):
    return df.rolling(window).apply(np.argmax) + 1

def ts_argmin(df, window=WINDOW):
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=PERIOD):
    if df.isnull().values.any():
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :].values
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    for row in range(period - 1, df.shape[0]):
        x = df.iloc[row - period + 1 : row + 1, :].values
        na_lwma[row, :] = np.dot(x.T, y)
    return pd.DataFrame(na_lwma, index=df.index, columns=["LWMA"])
