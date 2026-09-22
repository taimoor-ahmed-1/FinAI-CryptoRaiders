"""
Trading environment - replaces `TradeSimulator` from 5_erl_trainer / 6_erl_evaluator.

State (brief section 3.4 a):
    [position level, time in position] + 8 RNN factors + chosen LLM signals
    A  price-only          dim 10
    B  + sentiment_score   dim 11
    C  + risk_score        dim 11
    D  + both              dim 12   (the full model)

Actions (same as the notebook): 0 / 1 / 2 = decrease / keep / increase the position
level by one step, levels in [-max_position, +max_position] (no shorts if
allow_short=False). Level L means L / max_position x position_frac of current
equity is held in BTC (negative = short). Units only change when the level
changes, so an unchanged position is not re-balanced (and pays no cost). The size
is set so the position equals that fraction of equity after the trade's cost
(fully long = all equity in BTC, cash exactly 0).

Execution and accounting
    - the agent observes bar i and trades at the midpoint of bar i + exec_lag
      (default 1: the midpoint is the bar's (high+low)/2, only known after the bar);
    - every trade pays (fee_rate + slippage) x traded notional;
    - equity is marked at the execution price of the next decision; reward and
      metrics are therefore net of all costs.

Fixes relative to the notebook's TradeSimulator (see CHANGES.md for evidence)
    - one decision per environment copy, each at its own point in time (the notebook
      applied sim 0's action to all 64 copies and learned only from sim 0);
    - the state is read at the CURRENT bar (the training and evaluation loops read
      it at the episode's start bar, so it never changed within an episode);
    - episodes last `episode_bars` (default 3 days) instead of 2 decisions
      (seq_len 64 - 60 ignored steps, // step_gap 2);
    - realistic costs (notebook: 7e-7 slippage, no fee);
    - no hidden stop-loss: the notebook's `stop_loss_thresh = 1e-3` was in DOLLARS,
      so any position closed at the first $0.001 tick against it. A relative
      trailing stop is available (`stop_loss=0.02` = 2 %) but off by default;
    - LLM signals are scaled with the fixed 1-5 scale, (x - 1) / 4, instead of
      min-max over whichever split is loaded (the evaluator rescaled on the
      evaluation split itself);
    - signals come from the previous completed day (see prepare.py).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch as th

from .rewards import make_reward

STATE_CONFIGS = {
    "A": (),
    "B": ("sentiment_score",),
    "C": ("risk_score",),
    "D": ("sentiment_score", "risk_score"),
}
# fixed scaling for each signal column -> roughly [0, 1]
SIGNAL_SCALE = {
    "sentiment_score": (1.0, 5.0),
    "risk_score": (1.0, 5.0),
    "sentiment_weighted": (1.0, 5.0),
    "risk_weighted": (1.0, 5.0),
    "tweet_volume": "log1p",
}
BARS_PER_YEAR = 365 * 24 * 60          # 1-minute bars, crypto trades 24/7


@dataclass
class EnvConfig:
    state: str = "D"                    # A / B / C / D (STATE_CONFIGS) - ignored if `signals` is set
    signals: tuple = None               # explicit signal columns, e.g. ("sentiment_weighted",)
    signal_source: str = "lagged"       # "lagged" (causal, default) | "sameday" (dataset alignment - leak study only)
    step_gap: int = 5                   # bars between decisions
    episode_bars: int = 4320            # training episode length in bars (3 days)
    start_skip: int = 1024              # training episodes never start in the first bars (factor warm-up)
    exec_lag: int = 1
    max_position: int = 1
    allow_short: bool = True
    position_frac: float = 1.0
    fee_rate: float = 0.001             # Binance spot taker, per side
    slippage: float = 0.0002            # per side
    stop_loss: float = 0.0              # relative trailing stop, 0 = off
    max_holding_bars: int = 0           # force exit after this many bars, 0 = off
    holding_norm_bars: int = 1440       # "time in position" feature = bars / this, capped at 1
    reward: str = "net_asset_change"
    reward_scale: float = None          # None = the reward's own default
    reward_kwargs: dict = field(default_factory=dict)

    @property
    def signal_cols(self):
        return tuple(self.signals) if self.signals is not None else STATE_CONFIGS[self.state]

    @property
    def cost_rate(self):
        return self.fee_rate + self.slippage

    @property
    def state_dim(self):
        return 2 + 8 + len(self.signal_cols)

    @property
    def num_levels(self):
        return 2 * self.max_position + 1 if self.allow_short else self.max_position + 1

    @property
    def periods_per_year(self):
        return BARS_PER_YEAR / self.step_gap


def _scale_signal(col, x):
    rule = SIGNAL_SCALE.get(col)
    if rule == "log1p":
        return np.log1p(np.maximum(x, 0)) / 10.0
    if rule is None:
        return x
    lo, hi = rule
    return (x - lo) / (hi - lo)


class Market:
    """One split's prices, RNN factors and LLM signals, on the compute device."""

    def __init__(self, prepared_dir, factor_dir, split, cfg, device):
        m = np.load(Path(prepared_dir) / f"market_{split}.npz")
        factors = np.load(Path(factor_dir) / f"factors_{split}.npy").astype(np.float32)
        price = m["midpoint"].astype(np.float64)
        if len(factors) != len(price):
            raise ValueError(f"{split}: {len(factors)} factor rows vs {len(price)} price rows")
        cols = []
        for c in cfg.signal_cols:
            key = c if cfg.signal_source == "lagged" else f"sameday__{c}"
            if key not in m.files:
                raise KeyError(f"{key!r} not in market_{split}.npz - re-run prepare")
            cols.append(_scale_signal(c, m[key].astype(np.float32))[:, None])
        feat = np.concatenate([factors] + cols, axis=1) if cols else factors
        self.split = split
        self.n = len(price)
        self.price = th.tensor(price, device=device)
        self.feat = th.tensor(np.nan_to_num(feat), dtype=th.float32, device=device)
        self.time_ns = m["time_ns"]
        self.device = device


class TradingEnv:
    """Vectorised environment: `num_envs` independent copies stepped together.

    mode="train": each copy starts at a random bar and restarts after `episode_bars`.
    mode="eval":  the split is cut into `num_envs` consecutive chunks, copy k trades
                  chunk k once from its first bar (num_envs=1 = one exact pass).
    """

    def __init__(self, market, cfg, num_envs=1, mode="train", seed=0):
        self.m, self.cfg, self.n_env, self.mode = market, cfg, num_envs, mode
        self.dev = market.device
        self.gen = th.Generator(device="cpu").manual_seed(seed)
        self.state_dim, self.action_dim = cfg.state_dim, 3
        self.lo_level = -cfg.max_position if cfg.allow_short else 0
        self.reward_fn = make_reward(cfg.reward, num_envs, self.dev, cfg.reward_scale, **cfg.reward_kwargs)
        g, lag = cfg.step_gap, cfg.exec_lag
        if mode == "train":
            self.ep_steps = cfg.episode_bars // g
            self.lo_start = min(cfg.start_skip, market.n // 4)
            self.hi_start = market.n - 1 - lag - self.ep_steps * g
            if self.hi_start <= self.lo_start:
                raise ValueError(f"{market.split} split too short for episode_bars={cfg.episode_bars}")
        else:
            total = (market.n - 1 - lag) // g
            self.ep_steps = total // num_envs
            if self.ep_steps < 1:
                raise ValueError("evaluation split too short")
            self.chunk_start = th.arange(num_envs, device=self.dev) * self.ep_steps * g
        z = lambda dt=th.float64: th.zeros(num_envs, device=self.dev, dtype=dt)  # noqa: E731
        self.idx, self.t = z(th.long), z(th.long)
        self.cash, self.units, self.level = z(), z(), z(th.long)
        self.hold_bars, self.peak = z(th.long), z()

    # ------------------------------------------------------------------ helpers
    def _restart(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        if self.mode == "train":
            s = th.randint(self.lo_start, self.hi_start + 1, (k,), generator=self.gen).to(self.dev)
        else:
            s = self.chunk_start[mask]
        self.idx[mask] = s
        self.t[mask] = 0
        self.cash[mask] = 1.0
        self.units[mask] = 0.0
        self.level[mask] = 0
        self.hold_bars[mask] = 0
        self.peak[mask] = 0.0
        self.reward_fn.reset(mask)

    def obs(self):
        c = self.cfg
        pos = (self.level.float() / c.max_position)[:, None]
        held = (self.hold_bars.float() / c.holding_norm_bars).clamp(max=1.0)[:, None]
        return th.cat([pos, held, self.m.feat[self.idx]], dim=1)

    def equity(self, at_idx=None):
        i = self.idx if at_idx is None else at_idx
        return self.cash + self.units * self.m.price[i + self.cfg.exec_lag]

    # ------------------------------------------------------------------ API
    def reset(self):
        self._restart(th.ones(self.n_env, dtype=th.bool, device=self.dev))
        return self.obs()

    def step(self, action):
        """action: LongTensor (num_envs,) in {0, 1, 2}. Returns obs, reward, done, info.

        `done` marks episode ends (always time limits - there is no terminal state);
        info["final_obs"] holds the observation at that end, for bootstrapping.
        """
        c = self.cfg
        a = action.to(self.dev).long() - 1
        price = self.m.price
        p_now = price[self.idx]
        p_exec = price[self.idx + c.exec_lag]
        eq_prev = self.cash + self.units * p_exec

        level_prev = self.level
        new_level = (level_prev + a).clamp(self.lo_level, c.max_position)

        in_pos = level_prev != 0
        if c.stop_loss > 0:
            long_, short_ = level_prev > 0, level_prev < 0
            self.peak = th.where(long_, th.maximum(self.peak, p_now),
                                 th.where(short_, th.minimum(self.peak, p_now), self.peak))
            hit = (long_ & (p_now < self.peak * (1 - c.stop_loss))) | \
                  (short_ & (p_now > self.peak * (1 + c.stop_loss)))
            new_level = th.where(hit, th.zeros_like(new_level), new_level)
        if c.max_holding_bars > 0:
            new_level = th.where(in_pos & (self.hold_bars >= c.max_holding_bars),
                                 th.zeros_like(new_level), new_level)

        changed = new_level != level_prev
        # size the new position as a fraction of equity AFTER this trade's cost:
        # u1 = L * (E - cr*|u1 - u0|*p) / p, solved exactly for the trade direction s
        frac = new_level.double() / c.max_position * c.position_frac
        cr = c.cost_rate
        s = th.sign(frac * eq_prev / p_exec - self.units)
        target_units = frac * (eq_prev / p_exec + cr * s * self.units) / (1.0 + frac * cr * s)
        d_units = th.where(changed, target_units - self.units, th.zeros_like(self.units))
        traded = d_units.abs() * p_exec
        cost = traded * c.cost_rate
        self.cash = self.cash - d_units * p_exec - cost
        self.units = self.units + d_units

        opened = changed & (new_level != 0) & (th.sign(new_level) != th.sign(level_prev))
        self.peak = th.where(opened, p_exec, self.peak)
        self.hold_bars = th.where((new_level == 0) | opened, th.zeros_like(self.hold_bars),
                                  self.hold_bars) + th.where(new_level != 0, c.step_gap, 0)
        self.level = new_level

        self.idx = self.idx + c.step_gap
        self.t = self.t + 1
        eq_next = self.equity()
        reward = self.reward_fn(eq_prev, eq_next, level_prev.double(), new_level.double()).float()

        done = self.t >= self.ep_steps
        info = {"eq_prev": eq_prev, "eq_next": eq_next, "traded": traded, "cost": cost,
                "level": new_level}
        if bool(done.any()):
            info["final_obs"] = self.obs()
            if self.mode == "train":
                self._restart(done)
        return self.obs(), reward, done, info


@th.no_grad()
def run_policy(policy, market, cfg, chunks=1, seed=0):
    """Trade a whole split once with `policy(obs) -> actions` and record everything.

    chunks=1 is the exact, sequential backtest used for reported numbers. chunks>1
    trades `chunks` consecutive segments in parallel (each starting flat) - fast and
    good enough for choosing checkpoints on validation.

    Returns a dict of NumPy arrays, one entry per decision step, chunks stitched in
    time order: returns (net step return), level, traded (fraction of equity),
    cost (fraction of equity), time_ns; plus equity (with a leading 1.0).
    """
    env = TradingEnv(market, cfg, num_envs=chunks, mode="eval", seed=seed)
    obs = env.reset()
    T = env.ep_steps
    rec = {k: np.empty((T, chunks)) for k in ("eq_prev", "eq_next", "level", "traded", "cost")}
    bar = np.empty((T, chunks), dtype=np.int64)
    for t in range(T):
        bar[t] = env.idx.cpu().numpy()
        obs, _, _, info = env.step(policy(obs))
        for k in rec:
            rec[k][t] = info[k].double().cpu().numpy()
    ret = rec["eq_next"] / rec["eq_prev"] - 1.0
    flat = lambda a: a.T.reshape(-1)                                       # noqa: E731  chunk-major = time order
    out = {"returns": flat(ret), "level": flat(rec["level"]).astype(np.int8),
           "traded": flat(rec["traded"] / rec["eq_prev"]), "cost": flat(rec["cost"] / rec["eq_prev"]),
           "time_ns": market.time_ns[flat(bar)]}
    out["equity"] = np.concatenate([[1.0], np.cumprod(1.0 + out["returns"])])
    return out
