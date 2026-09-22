"""
Alpha101 technical factors (Kakushadze, 2016) for 1-minute BTC limit-order-book bars.

Provenance
----------
`TechIndicator` - the 101 factor formulas - is copied VERBATIM from the provided
notebook `3_alpha_signals_generator.ipynb`. Nothing in it has been edited.

Changes versus the notebook (all in the helper operators and the driver):

1. LEAK FIX - `decay_linear` no longer back-fills. The notebook called
   `df.bfill()`, which filled leading gaps with *future* values. Gaps are now
   forward-filled and any remaining leading gap is set to 0.
2. LEAK FIX - normalisation is fitted on the TRAIN split only and then applied
   to validation and test (`fit_quantiles` / `apply_quantiles`). The notebook
   fitted it separately on each split, so the test set was scaled with
   test-period statistics.
3. LEAK-FREE BY CONSTRUCTION - factors are computed once on the full
   chronological series and then sliced into splits. Every operator is
   backward-looking, so this introduces no look-ahead, and it removes the
   artificial warm-up gap the notebook created at each split boundary.
4. SPEED ONLY - `rank`, `ts_rank`, `product`, `ts_argmax`, `ts_argmin` and
   `decay_linear` are vectorised with NumPy sliding windows instead of calling
   a Python function per window. `tests/test_alpha101_equivalence.py` checks
   they match the notebook implementations exactly (NaNs and ties included).
   On 1.28M rows the notebook versions take hours; these take minutes.

The original helpers are kept in `_reference_helpers` for those tests.
"""
import time

import numpy as np
import pandas as pd
from numpy import abs, log, sign  # noqa: F401  (used inside the verbatim formulas)
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata  # noqa: F401

WINDOW = 10
PERIOD = 10


# --------------------------------------------------------------------------
# helper operators (unchanged from the notebook unless marked VECTORISED)
# --------------------------------------------------------------------------
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


def ts_min(df, window=WINDOW):
    return df.rolling(window).min()


def ts_max(df, window=WINDOW):
    return df.rolling(window).max()


def df_delta(df, period=1):
    return df.diff(period)


def delay(df, period=1):
    return df.shift(period)


def scale(df, window_size=WINDOW, k=1):
    scaled = df.mul(k)
    rolling_sums = np.abs(df).rolling(window=window_size, min_periods=1).sum()
    normalized = scaled.div(rolling_sums)
    return normalized


def _windows(obj, window):
    """(values, sliding windows, index, name, is_frame) for a Series or 1-col DataFrame.

    Row i of the window matrix holds rows [i, i+window) of the input, i.e. the
    window ENDING at row i+window-1 - exactly what pandas `rolling` uses.
    """
    is_frame = isinstance(obj, pd.DataFrame)
    s = obj.iloc[:, 0] if is_frame else obj
    vals = np.asarray(s, dtype=np.float64)
    # pandas' rolling machinery treats +/-inf as missing, so any window containing
    # an infinity yields NaN. Replicate that exactly (real bars do produce infs).
    vals = np.where(np.isinf(vals), np.nan, vals)
    win = sliding_window_view(vals, window) if len(vals) >= window else np.empty((0, window))
    return vals, win, s.index, s.name, is_frame


def _emit(out_tail, n, window, index, name, is_frame, template):
    """Place per-window results at their window-end rows; earlier rows are NaN."""
    out = np.full(n, np.nan)
    if len(out_tail):
        out[window - 1:] = out_tail
    res = pd.Series(out, index=index, name=name)
    return res.to_frame(template.columns[0]) if is_frame else res


def _tie_rank_of_last(win):
    """Average rank (1-based) of each window's last element, as scipy/pandas compute it."""
    last = win[:, -1][:, None]
    less = (win < last).sum(axis=1)
    equal = (win == last).sum(axis=1)
    return less + (equal + 1) / 2.0


def rolling_rank(na):
    return rankdata(na)[-1]


def rolling_prod(na):
    return np.prod(na)


def ts_rank(df, window=WINDOW):
    """VECTORISED. Same as notebook: rolling(window).apply(lambda na: rankdata(na)[-1])."""
    vals, win, idx, name, is_frame = _windows(df, window)
    r = _tie_rank_of_last(win)
    r[np.isnan(win).any(axis=1)] = np.nan
    return _emit(r, len(vals), window, idx, name, is_frame, df)


def product(df, window=WINDOW):
    """VECTORISED. Same as notebook: rolling(window).apply(np.prod)."""
    vals, win, idx, name, is_frame = _windows(df, window)
    p = win.prod(axis=1)
    p[np.isnan(win).any(axis=1)] = np.nan
    return _emit(p, len(vals), window, idx, name, is_frame, df)


def rank(df, window_size=WINDOW):
    """VECTORISED. Same as notebook: rolling(w).apply(lambda x: x.rank(pct=True).iloc[-1])."""
    vals, win, idx, name, is_frame = _windows(df, window_size)
    r = _tie_rank_of_last(win) / window_size
    r[np.isnan(win).any(axis=1)] = np.nan
    return _emit(r, len(vals), window_size, idx, name, is_frame, df)


def ts_argmax(df, window=WINDOW):
    """VECTORISED. Same as notebook: rolling(window).apply(np.argmax) + 1."""
    vals, win, idx, name, is_frame = _windows(df, window)
    a = win.argmax(axis=1).astype(np.float64) + 1
    a[np.isnan(win).any(axis=1)] = np.nan
    return _emit(a, len(vals), window, idx, name, is_frame, df)


def ts_argmin(df, window=WINDOW):
    """VECTORISED. Same as notebook: rolling(window).apply(np.argmin) + 1."""
    vals, win, idx, name, is_frame = _windows(df, window)
    a = win.argmin(axis=1).astype(np.float64) + 1
    a[np.isnan(win).any(axis=1)] = np.nan
    return _emit(a, len(vals), window, idx, name, is_frame, df)


def decay_linear(df, period=PERIOD):
    """FASTER + LEAK FIX. Linearly-weighted moving average (newest weight largest).

    Notebook behaviour kept: gaps are filled in place on the caller's frame, the
    first `period - 1` rows pass through unchanged. Removed: `bfill`, which
    copied future values into leading gaps.

    The per-row `np.dot(x.T, y)` is kept deliberately - it is the notebook's exact
    floating-point operation. A vectorised convolution sums in a different order;
    the results agree to ~1e-16, but several factors then rank the output with
    exact tie detection, where a 1-ulp difference turns a tie into a non-tie.
    The speed-up comes from slicing NumPy arrays instead of calling pandas
    `iloc` once per row.

    Output dtype follows the INPUT, as in the notebook (`np.zeros_like(df)`). For a
    boolean input (alpha092's first term) every weighted average is therefore
    stored as True/False - i.e. "was the condition true at least once in the
    window". That is a quirk of the original, kept so the factor is unchanged.
    """
    if df.isnull().values.any():
        df.ffill(inplace=True)
        df.fillna(value=0, inplace=True)   # was: df.bfill(inplace=True) first - look-ahead
    vals = df.values
    na_lwma = np.zeros_like(vals)          # same dtype as the notebook's np.zeros_like(df)
    na_lwma[:period, :] = vals[:period, :]
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    for row in range(period - 1, vals.shape[0]):
        x = vals[row - period + 1: row + 1, :]
        na_lwma[row, :] = np.dot(x.T, y)
    return pd.DataFrame(na_lwma, index=df.index, columns=["LWMA"])


class TechIndicator:
    def __init__(self, df):
        # Handle different column names for aggregated data
        if 'midpoint' in df.columns:
            self.vwap = df["midpoint"]
        elif 'price' in df.columns:
            self.vwap = df["price"]
        elif 'close' in df.columns:
            self.vwap = df["close"]
        else:
            raise ValueError("DataFrame must contain 'midpoint', 'price', or 'close' column")

        if 'spread' in df.columns:
            self.spread = df["spread"]
        else:
            self.spread = pd.Series(0.001, index=df.index)  # Default spread

        if 'sells' in df.columns:
            self.num_asks = df["sells"]
        elif 'volume' in df.columns:
            self.num_asks = df["volume"] / 2
        else:
            self.num_asks = pd.Series(1, index=df.index)

        if 'buys' in df.columns:
            self.num_bids = df["buys"]
        elif 'volume' in df.columns:
            self.num_bids = df["volume"] / 2
        else:
            self.num_bids = pd.Series(1, index=df.index)

        # Handle order book data if available
        if 'bids_distance_3' in df.columns and 'asks_distance_3' in df.columns:
            self.best_bid = self.vwap * (1 + df["bids_distance_3"])
            self.best_ask = self.vwap * (1 + df["asks_distance_3"])
        else:
            # Use simple bid/ask approximation
            self.best_bid = self.vwap * 0.999
            self.best_ask = self.vwap * 1.001

        if 'bids_notional_3' in df.columns and 'asks_notional_3' in df.columns:
            self.bid_volume = df["bids_notional_3"]
            self.ask_volume = df["asks_notional_3"]
        else:
            self.bid_volume = self.num_bids
            self.ask_volume = self.num_asks

        self.mid_price = (self.best_bid + self.best_ask) / 2
        self.returns = np.log(self.mid_price / self.mid_price.shift(1))
        self.volume = (self.num_asks + self.num_bids) / 2

    # Alpha functions (first 10 as example - add all 101 from alpha101_complete.py)
    def alpha001(self):
        inner = self.mid_price
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner**2, 5))

    def alpha002(self):
        df = -1 * correlation(
            rank(df_delta(log(self.bid_volume + 1), 2)),
            rank((self.mid_price - self.best_bid) / self.best_bid),
            6,
        )
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha003(self):
        df = -1 * correlation(rank(self.best_ask), rank(self.ask_volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha004(self):
        return -1 * ts_rank(rank(self.best_ask), 9)

    def alpha005(self):
        return rank((self.best_bid - (ts_sum(self.vwap, 10) / 10))) * (-1 * abs(rank((self.mid_price - self.vwap))))

    def alpha006(self):
        df = -1 * correlation(self.best_bid, self.bid_volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha007(self):
        adv20 = sma(self.bid_volume, 20)
        alpha = -1 * ts_rank(abs(df_delta(self.best_bid, 7)), 60) * sign(df_delta(self.best_bid, 7))
        alpha[adv20 >= self.bid_volume] = -1
        return alpha

    def alpha008(self):
        return -1 * (
            rank(
                (
                    (ts_sum(self.best_bid, 5) * ts_sum(self.returns, 5))
                    - delay((ts_sum(self.best_bid, 5) * ts_sum(self.returns, 5)), 10)
                )
            )
        )

    def alpha009(self):
        delta_midprice = df_delta(self.mid_price, 1)
        cond_1 = ts_min(delta_midprice, 5) > 0
        cond_2 = ts_max(delta_midprice, 5) < 0
        alpha = -1 * delta_midprice
        alpha[cond_1 | cond_2] = delta_midprice
        return alpha

    def alpha010(self):
        delta_midprice = df_delta(self.mid_price, 1)
        cond_1 = ts_min(delta_midprice, 4) > 0
        cond_2 = ts_max(delta_midprice, 4) < 0
        alpha = -1 * delta_midprice
        alpha[cond_1 | cond_2] = delta_midprice
        return alpha

    def alpha011(self):
        return (rank(ts_max((self.vwap - self.mid_price), 3)) + rank(ts_min((self.vwap - self.mid_price), 3))) * rank(
            df_delta(self.ask_volume, 3)
        )

    def alpha012(self):
        return sign(df_delta(self.bid_volume, 1)) * (-1 * df_delta(self.mid_price, 1))

    def alpha013(self):
        return -1 * rank(covariance(rank(self.mid_price), rank(self.ask_volume), 5))

    def alpha014(self):
        df = correlation(self.best_bid, self.bid_volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(df_delta(self.returns, 3)) * df

    def alpha015(self):
        df = correlation(rank(self.best_ask), rank(self.ask_volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.mid_price), rank(self.bid_volume), 5))

    def alpha017(self):
        adv20 = sma(self.bid_volume, 20)
        return -1 * (
            rank(ts_rank(self.mid_price, 10))
            * rank(df_delta(df_delta(self.mid_price, 1), 1))
            * rank(ts_rank((self.bid_volume / adv20), 5))
        )

    def alpha018(self):
        df = correlation(self.mid_price, self.best_bid, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.mid_price - self.best_bid)), 5) + (self.mid_price - self.best_bid)) + df))

    def alpha019(self):
        return (-1 * sign((self.mid_price - delay(self.mid_price, 7)) + df_delta(self.mid_price, 7))) * (
            1 + rank(1 + ts_sum(self.returns, 250))
        )

    def alpha020(self):
        return -1 * (
            rank(self.best_bid - delay(self.best_ask, 1))
            * rank(self.best_bid - delay(self.mid_price, 1))
            * rank(self.best_ask - delay(self.best_bid, 1))
        )

    def alpha021(self):
        cond_1 = sma(self.mid_price, 8) + stddev(self.mid_price, 8) < sma(self.mid_price, 2)
        cond_2 = sma(self.ask_volume, 20) / self.ask_volume < 1
        alpha = pd.DataFrame(np.ones_like(self.mid_price), index=self.mid_price.index)
        alpha[cond_1 | cond_2] = -1
        return alpha.squeeze(1)

    def alpha022(self):
        df = correlation(self.best_ask, self.ask_volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df_delta(df, 5) * rank(stddev(self.mid_price, 20))

    def alpha023(self):
        cond = sma(self.best_ask, 20) < self.best_ask
        alpha = pd.DataFrame(
            np.zeros_like(self.mid_price),
            index=self.mid_price.index,
            columns=["midprice"],
        )
        alpha.loc[cond, "midprice"] = -1 * df_delta(self.best_ask, 2).fillna(value=0)
        return alpha.squeeze(1)

    def alpha024(self):
        cond = df_delta(sma(self.mid_price, 100), 100) / delay(self.mid_price, 100) <= 0.05
        alpha = -1 * df_delta(self.mid_price, 3)
        alpha[cond] = -1 * (self.mid_price - ts_min(self.mid_price, 100))
        return alpha

    def alpha025(self):
        adv20 = sma(self.ask_volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.best_ask - self.mid_price)))

    def alpha026(self):
        df = correlation(ts_rank(self.bid_volume, 5), ts_rank(self.best_bid, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)

    def alpha027(self):
        alpha = rank((sma(correlation(rank(self.bid_volume), rank(self.best_bid), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha

    def alpha028(self):
        adv20 = sma(self.bid_volume, 20)
        df = correlation(adv20, self.best_bid, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(df)

    def alpha029(self):
        delta_mid = df_delta((self.mid_price - 1), 5)
        rank_neg_delta = rank(-1 * delta_mid)
        sum_ranks = ts_sum(rank(rank_neg_delta), 2)
        scaled_log = scale(log(sum_ranks + 1))
        min_rank = ts_min(rank(rank(scaled_log)), 5)
        delayed_ret = delay(-1 * self.returns, 6)
        ts_rank_ret = ts_rank(delayed_ret, 5)
        return min_rank + ts_rank_ret

    def alpha030(self):
        delta_midprice = df_delta(self.mid_price, 1)
        inner = sign(delta_midprice) + sign(delay(delta_midprice, 1)) + sign(delay(delta_midprice, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.ask_volume, 5)) / ts_sum(self.ask_volume, 20)

    def alpha031(self):
        adv20 = sma(self.bid_volume, 20)
        df = correlation(adv20, self.best_bid, 12).replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(df_delta(self.mid_price, 10)))).to_frame(), 10))))
        p2 = rank((-1 * df_delta(self.mid_price, 3)))
        p3 = sign(scale(df))
        return p1.LWMA + p2 + p3

    def alpha032(self):
        return scale(((sma(self.mid_price, 7) / 7) - self.mid_price)) + (
            20 * scale(correlation(self.vwap, delay(self.mid_price, 5), 230))
        )

    def alpha033(self):
        return rank(-1 + (self.best_ask / self.mid_price))

    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(df_delta(self.mid_price, 1)))

    def alpha035(self):
        return (ts_rank(self.bid_volume, 32) * (1 - ts_rank(self.mid_price + self.spread, 16))) * (
            1 - ts_rank(self.returns, 32)
        )

    def alpha036(self):
        adv20 = sma(self.bid_volume, 20)
        corr_mid_bid = correlation(self.mid_price - self.best_bid, delay(self.bid_volume, 1), 15)
        rank_corr_mid_bid = rank(corr_mid_bid)
        rank_bid_mid = rank(self.best_bid - self.mid_price)
        delayed_neg_ret = delay(-1 * self.returns, 6)
        ts_rank_neg_ret = ts_rank(delayed_neg_ret, 5)
        rank_ts_rank_neg_ret = rank(ts_rank_neg_ret)
        abs_corr_vwap_adv20 = abs(correlation(self.vwap, adv20, 6))
        rank_abs_corr_vwap_adv20 = rank(abs_corr_vwap_adv20)
        sma_mid_200 = sma(self.mid_price, 200) / 200
        diff_sma_mid_bid = sma_mid_200 - self.best_bid
        prod_diff_mid_bid = diff_sma_mid_bid * (self.mid_price - self.best_bid)
        rank_prod_diff_mid_bid = rank(prod_diff_mid_bid)
        result = (
            (2.21 * rank_corr_mid_bid)
            + (0.7 * rank_bid_mid)
            + (0.73 * rank_ts_rank_neg_ret)
            + rank_abs_corr_vwap_adv20
            + (0.6 * rank_prod_diff_mid_bid)
        )
        return result

    def alpha037(self):
        return rank(correlation(delay(self.best_ask - self.mid_price, 1), self.mid_price, 200)) + rank(
            self.best_ask - self.mid_price
        )

    def alpha038(self):
        inner = self.mid_price / self.best_ask
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.best_ask, 10)) * rank(inner)

    def alpha039(self):
        adv20 = sma(self.ask_volume, 20)
        return (
            -1
            * rank(
                df_delta(self.mid_price, 7) * (1 - rank(decay_linear((self.ask_volume / adv20).to_frame(), 9).LWMA))
            )
        ) * (1 + rank(sma(self.returns, 250)))

    def alpha040(self):
        return -1 * rank(stddev(self.best_bid, 10)) * correlation(self.best_bid, self.bid_volume, 10)

    def alpha041(self):
        return pow((self.best_bid * self.best_ask), 0.5) - self.vwap

    def alpha042(self):
        return rank((self.vwap - self.mid_price)) / rank((self.vwap + self.mid_price))

    def alpha043(self):
        adv20 = sma(self.ask_volume, 20)
        return ts_rank(self.ask_volume / adv20, 20) * ts_rank((-1 * df_delta(self.mid_price, 7)), 8)

    def alpha044(self):
        df = correlation(self.best_ask, rank(self.ask_volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    def alpha045(self):
        df = correlation(self.mid_price, self.ask_volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (
            rank(sma(delay(self.mid_price, 5), 20))
            * df
            * rank(correlation(ts_sum(self.mid_price, 5), ts_sum(self.mid_price, 20), 2))
        )

    def alpha046(self):
        inner = ((delay(self.mid_price, 20) - delay(self.mid_price, 10)) / 10) - (
            (delay(self.mid_price, 10) - self.mid_price) / 10
        )
        alpha = -1 * df_delta(self.mid_price)
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    def alpha047(self):
        adv20 = sma(self.bid_volume, 20)
        return (
            ((rank((1 / self.mid_price)) * self.bid_volume) / adv20)
            * ((self.best_bid * rank((self.best_bid - self.mid_price))) / (sma(self.best_bid, 5) / 5))
        ) - rank((self.vwap - delay(self.vwap, 5)))

    def alpha048(self):
        adv20 = sma(self.bid_volume, 20)
        return (
            ((rank((1 / self.mid_price)) * self.ask_volume) / adv20)
            * ((self.best_ask * rank((self.best_ask - self.mid_price))) / (sma(self.best_ask, 5) / 5))
        ) - rank((self.vwap - delay(self.vwap, 5)))

    def alpha049(self):
        inner = ((delay(self.mid_price, 20) - delay(self.mid_price, 10)) / 10) - (
            (delay(self.mid_price, 10) - self.mid_price) / 10
        )
        alpha = -1 * df_delta(self.mid_price)
        alpha[inner < -0.1] = 1
        return alpha

    def alpha050(self):
        return -1 * ts_max(rank(correlation(rank(self.ask_volume), rank(self.vwap), 5)), 5)

    def alpha051(self):
        inner = ((delay(self.mid_price, 20) - delay(self.mid_price, 10)) / 10) - (
            (delay(self.mid_price, 10) - self.mid_price) / 10
        )
        alpha = -1 * df_delta(self.mid_price)
        alpha[inner < -0.05] = 1
        return alpha

    def alpha052(self):
        return (
            (-1 * df_delta(ts_min(self.best_bid, 5), 5))
            * rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))
        ) * ts_rank(self.bid_volume, 5)

    def alpha053(self):
        inner = self.spread.replace(0, 0.0001)
        return -1 * df_delta(
            (((self.mid_price - self.best_bid) - (self.best_bid - self.mid_price)) / inner),
            9,
        )

    def alpha054(self):
        inner = self.spread.replace(0, -0.0001)
        return -1 * (self.best_bid - self.mid_price) * (self.best_ask**5) / (inner * (self.mid_price**5))

    def alpha055(self):
        divisor = (ts_max(self.best_ask, 12) - ts_min(self.best_bid, 12)).replace(0, 0.0001)
        inner = (self.mid_price - ts_min(self.best_bid, 12)) / divisor
        df = correlation(rank(inner), rank(self.bid_volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha056(self):
        inner = self.spread.replace(0, -0.0001)
        return -1 * (self.best_ask - self.mid_price) * (self.best_bid**5) / (inner * (self.mid_price**5))

    def alpha057(self):
        return 0 - (
            1 * ((self.mid_price - self.vwap) / decay_linear(rank(ts_argmax(self.mid_price, 30)).to_frame(), 2).LWMA)
        )

    def alpha058(self):
        return sign(df_delta(self.ask_volume, 1)) * (-1 * df_delta(self.mid_price, 1))

    def alpha059(self):
        divisor = self.spread.replace(0, 0.0001)
        inner = ((self.mid_price - self.best_ask) - (self.best_ask - self.mid_price)) * self.ask_volume / divisor
        return -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.mid_price, 10))))

    def alpha060(self):
        divisor = self.spread.replace(0, 0.0001)
        inner = ((self.mid_price - self.best_bid) - (self.best_bid - self.mid_price)) * self.bid_volume / divisor
        return -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.mid_price, 10))))

    def alpha061(self):
        adv180 = sma(self.ask_volume, 180)
        return rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18))

    def alpha062(self):
        adv20 = sma(self.ask_volume, 20)
        value1 = rank(correlation(self.vwap, sma(adv20, 22), 10))
        value2 = rank(self.best_ask) + rank(self.best_ask)
        value3 = rank(((self.best_ask + self.best_bid) / 2)) + rank(self.best_ask)
        return value1 - rank(value2 - value3)

    def alpha063(self):
        adv120 = sma(self.bid_volume, 120)
        sma_bid_ask = sma(((self.best_bid * 0.178404) + (self.best_ask * (1 - 0.178404))), 13)
        sma_adv120 = sma(adv120, 13)
        corr = correlation(sma_bid_ask, sma_adv120, 17)
        rank_corr = rank(corr)
        weighted_price = (self.mid_price * 0.178404) + (self.vwap * (1 - 0.178404))
        delta_weighted_price = df_delta(weighted_price, 3)
        rank_delta = rank(delta_weighted_price)
        return rank_corr - rank_delta

    def alpha064(self):
        adv120 = sma(self.ask_volume, 120)
        return (
            rank(
                correlation(
                    sma(
                        ((self.best_ask * 0.178404) + (self.best_bid * (1 - 0.178404))),
                        13,
                    ),
                    sma(adv120, 13),
                    17,
                )
            )
            < rank(
                df_delta(
                    ((((self.best_ask + self.best_bid) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))),
                    3,
                )
            )
        ) * -1

    def alpha065(self):
        adv60 = sma(self.ask_volume, 60)
        return (
            rank(
                correlation(
                    ((self.best_ask * 0.00817205) + (self.vwap * (1 - 0.00817205))),
                    sma(adv60, 9),
                    6,
                )
            )
            < rank((self.best_ask - ts_min(self.best_ask, 14)))
        ) * -1

    def alpha066(self):
        vwap_delta = df_delta(self.vwap, 4).to_frame()
        decay_vwap_delta = decay_linear(vwap_delta, 7).LWMA
        rank_decay_vwap_delta = rank(decay_vwap_delta)
        bid_ask_diff = ((self.best_bid * 0.96633) + (self.best_bid * (1 - 0.96633))) - self.vwap
        price_ratio = (bid_ask_diff / (self.best_ask - self.best_bid)).to_frame()
        decay_price_ratio = decay_linear(price_ratio, 11).LWMA
        ts_rank_decay_price_ratio = ts_rank(decay_price_ratio, 7)
        result = (rank_decay_vwap_delta + ts_rank_decay_price_ratio) * -1
        return result

    def alpha067(self):
        adv15 = sma(self.bid_volume, 15)
        return (
            ts_rank(correlation(rank(self.best_bid), rank(adv15), 9), 14)
            < rank(df_delta(((self.mid_price * 0.518371) + (self.best_ask * (1 - 0.518371))), 3))
        ) * -1

    def alpha068(self):
        adv15 = sma(self.ask_volume, 15)
        return (
            ts_rank(correlation(rank(self.best_ask), rank(adv15), 9), 14)
            < rank(df_delta(((self.mid_price * 0.518371) + (self.best_bid * (1 - 0.518371))), 3))
        ) * -1

    def alpha069(self):
        adv180 = sma(self.bid_volume, 180)
        return rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18))

    def alpha070(self):
        adv60 = sma(self.bid_volume, 60)
        return (
            rank(
                correlation(
                    ((self.best_bid * 0.00817205) + (self.vwap * (1 - 0.00817205))),
                    sma(adv60, 9),
                    6,
                )
            )
            < rank((self.best_bid - ts_min(self.best_bid, 14)))
        ) * -1

    def alpha071(self):
        adv180 = sma(self.bid_volume, 180)
        p1 = ts_rank(
            decay_linear(
                correlation(ts_rank(self.mid_price, 3), ts_rank(adv180, 12), 18).to_frame(),
                4,
            ).LWMA,
            16,
        )
        p2 = ts_rank(
            decay_linear(
                (rank(((self.best_bid + self.best_ask) - (self.vwap + self.vwap))).pow(2)).to_frame(),
                16,
            ).LWMA,
            4,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "max"] = df["p1"]
        df.loc[df["p2"] >= df["p1"], "max"] = df["p2"]
        return df["max"]

    def alpha072(self):
        adv40 = sma(self.bid_volume, 40)
        return rank(decay_linear(correlation(self.mid_price, adv40, 9).to_frame(), 10).LWMA) / rank(
            decay_linear(
                correlation(ts_rank(self.vwap, 4), ts_rank(self.bid_volume, 19), 7).to_frame(),
                3,
            ).LWMA
        )

    def alpha073(self):
        p1 = rank(decay_linear(df_delta(self.vwap, 5).to_frame(), 3).LWMA)
        p2 = ts_rank(
            decay_linear(
                (
                    (
                        df_delta(
                            ((self.best_ask * 0.147155) + (self.best_bid * (1 - 0.147155))),
                            2,
                        )
                        / ((self.mid_price * 0.147155) + (self.best_bid * (1 - 0.147155)))
                    )
                    * -1
                ).to_frame(),
                3,
            ).LWMA,
            17,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "max"] = df["p1"]
        df.loc[df["p2"] >= df["p1"], "max"] = df["p2"]
        return -1 * df["max"]

    def alpha074(self):
        adv30 = sma(self.bid_volume, 30)
        return (
            rank(correlation(self.mid_price, sma(adv30, 37), 15))
            < rank(
                correlation(
                    rank(((self.best_bid * 0.0261661) + (self.vwap * (1 - 0.0261661)))),
                    rank(self.bid_volume),
                    11,
                )
            )
        ) * -1

    def alpha075(self):
        adv50 = sma(self.bid_volume, 50)
        return rank(correlation(self.vwap, self.bid_volume, 4)) < rank(
            correlation(rank(self.best_bid), rank(adv50), 12)
        )

    def alpha076(self):
        adv50 = sma(self.ask_volume, 50)
        return rank(correlation(self.vwap, self.ask_volume, 4)) < rank(
            correlation(rank(self.best_ask), rank(adv50), 12)
        )

    def alpha077(self):
        adv40 = sma(self.ask_volume, 40)
        p1 = rank(
            decay_linear(
                ((self.mid_price + self.best_ask) - (self.vwap + self.best_ask)).to_frame(),
                20,
            ).LWMA
        )
        p2 = rank(decay_linear(correlation(self.mid_price, adv40, 3).to_frame(), 6).LWMA)
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "min"] = df["p2"]
        df.loc[df["p2"] >= df["p1"], "min"] = df["p1"]
        return df["min"]

    def alpha078(self):
        adv40 = sma(self.bid_volume, 40)
        return rank(
            correlation(
                ts_sum(((self.best_bid * 0.352233) + (self.vwap * (1 - 0.352233))), 20),
                ts_sum(adv40, 20),
                7,
            )
        ).pow(rank(correlation(rank(self.vwap), rank(self.bid_volume), 6)))

    def alpha079(self):
        adv40 = sma(self.ask_volume, 40)
        return rank(
            correlation(
                ts_sum(((self.best_ask * 0.352233) + (self.vwap * (1 - 0.352233))), 20),
                ts_sum(adv40, 20),
                7,
            )
        ).pow(rank(correlation(rank(self.vwap), rank(self.ask_volume), 6)))

    def alpha080(self):
        adv10 = sma(self.bid_volume, 10)
        return (
            rank(
                log(
                    product(
                        rank((rank(correlation(self.vwap, ts_sum(adv10, 50), 8)).pow(4))),
                        15,
                    )
                )
            )
            < rank(correlation(rank(self.vwap), rank(self.bid_volume), 5))
        ) * -1

    def alpha081(self):
        adv10 = sma(self.ask_volume, 10)
        return (
            rank(
                log(
                    product(
                        rank((rank(correlation(self.vwap, ts_sum(adv10, 50), 8)).pow(4))),
                        15,
                    )
                )
            )
            < rank(correlation(rank(self.vwap), rank(self.ask_volume), 5))
        ) * -1

    def alpha082(self):
        adv20 = sma(self.bid_volume, 20)
        return (
            ts_rank(correlation(self.mid_price, sma(adv20, 15), 6), 20)
            < rank(((self.best_bid + self.mid_price) - (self.vwap + self.best_bid)))
        ) * -1

    def alpha083(self):
        return (rank(delay((self.spread / (ts_sum(self.mid_price, 5) / 5)), 2)) * rank(rank(self.bid_volume))) / (
            (self.spread / (ts_sum(self.mid_price, 5) / 5)) / (self.vwap - self.mid_price)
        )

    def alpha084(self):
        return pow(
            ts_rank((self.vwap - ts_max(self.vwap, 15)), 21),
            df_delta(self.mid_price, 5),
        )

    def alpha085(self):
        adv30 = sma(self.ask_volume, 30)
        return rank(
            correlation(
                ((self.best_ask * 0.876703) + (self.mid_price * (1 - 0.876703))),
                adv30,
                10,
            )
        ).pow(rank(correlation(ts_rank(self.mid_price, 4), ts_rank(self.ask_volume, 10), 7)))

    def alpha086(self):
        adv20 = sma(self.ask_volume, 20)
        return (
            ts_rank(correlation(self.mid_price, sma(adv20, 15), 6), 20)
            < rank(((self.best_ask + self.mid_price) - (self.vwap + self.best_ask)))
        ) * -1

    def alpha087(self):
        return -1 * ts_rank(rank(self.best_bid), 9)

    def alpha088(self):
        adv60 = sma(self.ask_volume, 60)
        rank_ask = rank(self.best_ask)
        rank_bid = rank(self.best_bid)
        rank_mid = rank(self.mid_price)
        diff_rank = (rank_ask + rank_bid) - (rank_ask + rank_mid)
        decay_diff_rank = decay_linear(diff_rank.to_frame(), 8).LWMA
        p1 = rank(decay_diff_rank)
        ts_rank_mid = ts_rank(self.mid_price, 8)
        ts_rank_adv60 = ts_rank(adv60, 21)
        corr_ts_rank = correlation(ts_rank_mid, ts_rank_adv60, 8)
        decay_corr_ts_rank = decay_linear(corr_ts_rank.to_frame(), 7).LWMA
        p2 = ts_rank(decay_corr_ts_rank, 3)
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df["min"] = df[["p1", "p2"]].min(axis=1)
        return df["min"]

    def alpha089(self):
        df = -1 * correlation(
            rank(df_delta(log(self.ask_volume + 1), 2)),
            rank((self.mid_price - self.best_ask) / self.best_ask),
            6,
        )
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha090(self):
        return -1 * (
            rank(
                (
                    (ts_sum(self.best_ask, 5) * ts_sum(self.returns, 5))
                    - delay((ts_sum(self.best_ask, 5) * ts_sum(self.returns, 5)), 10)
                )
            )
        )

    def alpha091(self):
        df = correlation(self.best_bid, self.bid_volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df_delta(df, 5) * rank(stddev(self.mid_price, 20))

    def alpha092(self):
        adv30 = sma(self.bid_volume, 30)
        p1 = ts_rank(
            decay_linear(
                ((self.mid_price + self.mid_price) < (self.best_ask + self.best_bid)).to_frame(),
                15,
            ).LWMA,
            19,
        )
        p2 = ts_rank(
            decay_linear(correlation(rank(self.best_bid), rank(adv30), 8).to_frame(), 7).LWMA,
            7,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "min"] = df["p2"]
        df.loc[df["p2"] >= df["p1"], "min"] = df["p1"]
        return df["min"]

    def alpha093(self):
        adv60 = sma(self.bid_volume, 60)
        return (
            rank((self.vwap - ts_min(self.vwap, 12))).pow(
                ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)
            )
            * -1
        )

    def alpha094(self):
        adv60 = sma(self.ask_volume, 60)
        return (
            rank((self.vwap - ts_min(self.vwap, 12))).pow(
                ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)
            )
            * -1
        )

    def alpha095(self):
        adv40 = sma(self.bid_volume, 40)
        return rank((self.best_bid - ts_min(self.best_bid, 12))) < ts_rank(
            (rank(correlation(sma(self.mid_price, 19), sma(adv40, 19), 13)).pow(5)), 12
        )

    def alpha096(self):
        adv60 = sma(self.ask_volume, 60)
        p1 = ts_rank(
            decay_linear(correlation(rank(self.vwap), rank(self.ask_volume).to_frame(), 4), 4).LWMA,
            8,
        )
        p2 = ts_rank(
            decay_linear(
                ts_argmax(correlation(ts_rank(self.mid_price, 7), ts_rank(adv60, 4), 4), 13).to_frame(),
                14,
            ).LWMA,
            13,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "max"] = df["p1"]
        df.loc[df["p2"] >= df["p1"], "max"] = df["p2"]
        return -1 * df["max"]

    def alpha097(self):
        adv5 = sma(self.ask_volume, 5)
        adv15 = sma(self.ask_volume, 15)
        return rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5).to_frame(), 7).LWMA) - rank(
            decay_linear(
                ts_rank(ts_argmin(correlation(rank(self.best_ask), rank(adv15), 21), 9), 7).to_frame(),
                8,
            ).LWMA
        )

    def alpha098(self):
        adv5 = sma(self.bid_volume, 5)
        adv15 = sma(self.bid_volume, 15)
        return rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5).to_frame(), 7).LWMA) - rank(
            decay_linear(
                ts_rank(ts_argmin(correlation(rank(self.best_bid), rank(adv15), 21), 9), 7).to_frame(),
                8,
            ).LWMA
        )

    def alpha099(self):
        adv60 = sma(self.bid_volume, 60)
        return (
            rank(correlation(ts_sum(self.mid_price, 20), ts_sum(adv60, 20), 9))
            < rank(correlation(self.best_bid, self.bid_volume, 6))
        ) * -1

    def alpha100(self):
        adv60 = sma(self.ask_volume, 60)
        return (
            rank(correlation(ts_sum(self.mid_price, 20), ts_sum(adv60, 20), 9))
            < rank(correlation(self.best_ask, self.ask_volume, 6))
        ) * -1

    def alpha101(self):
        return (self.mid_price - self.best_bid) / (self.spread + 0.001)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
N_ALPHAS = 101


def compute_raw_alphas(df, verbose=True):
    """All 101 factors on a chronological frame -> float64 array (rows, 101).

    NaN / inf are replaced by 0, as in the notebook. Kept in float64 until
    normalisation, as the notebook does: a few factors reach finite values beyond
    the float32 range, which would overflow to inf and poison the quantiles.
    No normalisation here - fit it on the training rows with `fit_quantiles`.
    """
    indicator = TechIndicator(df=df)
    out = np.zeros((len(df), N_ALPHAS), dtype=np.float64)
    t0 = time.time()
    for i in range(1, N_ALPHAS + 1):
        t1 = time.time()
        try:
            a = getattr(indicator, f"alpha{i:03}")()
            a = np.asarray(a.values if hasattr(a, "values") else a, dtype=np.float64).reshape(len(df), -1)[:, 0]
            out[:, i - 1] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:  # the notebook also fell back to zeros
            print(f"  alpha{i:03d} failed -> zeros ({e})")
        if verbose and (i % 10 == 0 or i == N_ALPHAS):
            print(f"  alpha {i:3d}/{N_ALPHAS}  last {time.time() - t1:5.1f}s  total {time.time() - t0:6.0f}s")
    return out


def fit_quantiles(train_alphas, q_low=0.01, q_high=0.99):
    """Per-factor clip bounds from the TRAIN rows only (same quantiles as the notebook)."""
    lo = np.quantile(train_alphas, q_low, axis=0, keepdims=True)
    hi = np.quantile(train_alphas, q_high, axis=0, keepdims=True)
    return lo, hi


def apply_quantiles(alphas, lo, hi):
    """Clip to [lo, hi] and map linearly to [-1, 1] - the notebook's transform, train-fitted."""
    a = np.clip(alphas, lo, hi)
    den = np.where((hi - lo) == 0, 1, hi - lo)
    out = (2 * (a - lo) / den - 1).astype(np.float16)
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite values after normalisation")
    return out
