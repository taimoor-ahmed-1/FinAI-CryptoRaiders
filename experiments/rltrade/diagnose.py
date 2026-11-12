"""
Post-hoc diagnosis of saved agents (experiment 18): what a policy did on a split, and why.

Nothing here trains, tunes or selects. Every function reads a saved run (runner.load_saved)
or a trace recorded from one, so the numbers explain results that are already fixed.

A trace is one exact backtest (env.run_policy, chunks=1) plus, per decision step:
    mret   the market's simple return over the step the position is held for
           (price[bar + exec_lag] -> price[bar + step_gap + exec_lag], as the environment marks it);
    held   the "time in position" input (capped at 1 = one day);
    q      the agent's Q-values for the three actions (DQN-family agents only).

Attribution (sums of step returns, in %): gross = level * mret, and
    gross = exposure + timing,  exposure = mean(level) * sum(mret),  timing = sum((level - mean(level)) * mret),
so "exposure" is what a constant position at the policy's average level would have earned and
"timing" is what the changes of position added. Net = gross - costs, up to a small residual.
"""
import numpy as np
import torch as th
from scipy import stats

from .env import Market, run_policy

TRACE_KEYS = ("returns", "level", "cost", "traded", "time_ns", "bar", "mret", "held", "q", "equity")


def trace(run_dir, split, prepared_dir, factor_dir, device="auto"):
    """Backtest a saved run on `split` once and record the extra per-step series above."""
    from .runner import load_saved
    policy, agent, cfg, _, dev = load_saved(run_dir, device)
    market = Market(prepared_dir, factor_dir, split, cfg, dev)
    held, qs = [], []
    has_q = agent is not None and hasattr(agent, "q")

    @th.no_grad()
    def pol(o):
        held.append(float(o[0, 1]))
        if has_q:
            q = agent.q(o.to(agent.device))
            qs.append(q[0].float().cpu().numpy())
            return q.argmax(-1)
        return policy(o)

    bt = run_policy(pol, market, cfg, chunks=1)
    price = market.price.cpu().numpy()
    b, g, lag = bt["bar"], cfg.step_gap, cfg.exec_lag
    bt["mret"] = price[b + g + lag] / price[b + lag] - 1.0
    bt["held"] = np.asarray(held, dtype=np.float32)
    bt["q"] = np.asarray(qs, dtype=np.float32) if has_q else np.zeros((len(b), 0), np.float32)
    return bt


def steps_per_day(step_gap):
    return int(round(1440 / step_gap))


def spells(level):
    """(sign, length in steps) of every run of a constant position sign."""
    s = np.sign(np.asarray(level))
    if len(s) == 0:
        return []
    cut = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(s)]])
    return [(int(s[a]), int(e - a)) for a, e in zip(starts, ends)]


def behaviour(tr, step_gap=5):
    lv = np.asarray(tr["level"])
    spd = steps_per_day(step_gap)
    sp = [n for sgn, n in spells(lv) if sgn != 0]
    changes = int(np.count_nonzero(np.diff(np.concatenate([[0], lv]))))
    hours = np.asarray(sp, float) * step_gap / 60.0
    return {"long_pct": 100 * float(np.mean(lv > 0)), "short_pct": 100 * float(np.mean(lv < 0)),
            "flat_pct": 100 * float(np.mean(lv == 0)), "trades": changes,
            "trades_per_day": changes / (len(lv) / spd), "positions": len(sp),
            "median_hold_h": float(np.median(hours)) if len(sp) else float("nan"),
            "longest_hold_d": float(hours.max() / 24) if len(sp) else float("nan"),
            "costs_pct": 100 * float(np.sum(tr["cost"])),
            "held_capped_pct": 100 * float(np.mean(np.asarray(tr["held"]) >= 1.0 - 1e-6))}


def attribution(tr, step_gap=5):
    lv = np.asarray(tr["level"], float)
    r, mr, c = (np.asarray(tr[k], float) for k in ("returns", "mret", "cost"))
    gross = lv * mr
    exposure = lv.mean() * mr.sum()
    out = {"net_pct": 100 * r.sum(), "gross_pct": 100 * gross.sum(), "exposure_pct": 100 * exposure,
           "timing_pct": 100 * (gross.sum() - exposure), "costs_pct": 100 * c.sum(),
           "from_long_pct": 100 * gross[lv > 0].sum(), "from_short_pct": 100 * gross[lv < 0].sum(),
           "residual_pct": 100 * (r.sum() - gross.sum() + c.sum()), "mean_level": lv.mean()}
    if mr.std() > 0 and len(r) > 2:
        fit = stats.linregress(mr, r)
        ppy = 365 * steps_per_day(step_gap)
        out.update(beta=fit.slope, alpha_ann_pct=100 * fit.intercept * ppy, r2=fit.rvalue ** 2)
    else:
        out.update(beta=float("nan"), alpha_ann_pct=float("nan"), r2=float("nan"))
    return out


def daily_timing(tr, step_gap=5):
    """Position at each day's first decision vs the market's return over the next day (non-overlapping).

    Returns the correlation, its p-value and the number of days; NaN when the position never
    changes between days (a static policy has no timing to measure).
    """
    spd = steps_per_day(step_gap)
    lv = np.asarray(tr["level"], float)
    lm = np.log1p(np.asarray(tr["mret"], float))
    n = len(lv) // spd
    if n < 3:
        return {"ic": float("nan"), "ic_p": float("nan"), "days": n, "hit_pct": float("nan")}
    pos = lv[: n * spd: spd][:n]
    fwd = lm[: n * spd].reshape(n, spd).sum(1)
    nz = pos != 0
    hit = 100 * float(np.mean(np.sign(pos[nz]) == np.sign(fwd[nz]))) if nz.any() else float("nan")
    if np.std(pos) == 0:
        return {"ic": float("nan"), "ic_p": float("nan"), "days": n, "hit_pct": hit}
    rho, p = stats.spearmanr(pos, fwd)
    return {"ic": float(rho), "ic_p": float(p), "days": n, "hit_pct": hit}


def q_gap(tr):
    """Margin between the best and second-best action's Q-value at each step (empty without Q-values)."""
    q = np.asarray(tr["q"])
    if q.ndim != 2 or q.shape[1] < 2:
        return np.zeros(0)
    s = np.sort(q, axis=1)
    return s[:, -1] - s[:, -2]


def factor_shift(f_ref, f_other, max_n=100_000, seed=0):
    """Per factor: KS statistic vs the reference split, mean shift in reference stds, share outside
    the reference's 0.5-99.5 % range; plus the share of rows with ANY factor outside it."""
    rng = np.random.default_rng(seed)
    a = np.asarray(f_ref, np.float64)
    b = np.asarray(f_other, np.float64)
    ia = rng.choice(len(a), min(max_n, len(a)), replace=False)
    ib = rng.choice(len(b), min(max_n, len(b)), replace=False)
    lo, hi = np.percentile(a, 0.5, axis=0), np.percentile(a, 99.5, axis=0)
    out = b < lo
    out |= b > hi
    rows = []
    for j in range(a.shape[1]):
        rows.append({"factor": j, "ks": float(stats.ks_2samp(a[ia, j], b[ib, j]).statistic),
                     "mean_shift_sd": float((b[:, j].mean() - a[:, j].mean()) / (a[:, j].std() + 1e-12)),
                     "sd_ratio": float(b[:, j].std() / (a[:, j].std() + 1e-12)),
                     "outside_pct": 100 * float(out[:, j].mean())})
    return rows, 100 * float(out.any(axis=1).mean())


def agreement(levels):
    """levels: (n_seeds, T). Mean pairwise share of steps with the same position sign, and the share
    of steps where a strict majority of seeds is long / short."""
    s = np.sign(np.asarray(levels))
    k = len(s)
    pairs = [np.mean(s[i] == s[j]) for i in range(k) for j in range(i + 1, k)]
    return {"pairwise_same_pct": 100 * float(np.mean(pairs)) if pairs else float("nan"),
            "majority_long_pct": 100 * float(np.mean((s > 0).sum(0) > k / 2)),
            "majority_short_pct": 100 * float(np.mean((s < 0).sum(0) > k / 2))}


def sharpe(r, ppy):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def bootstrap_sharpe(returns, ref_returns, step_gap=5, n_boot=2000, seed=0):
    """Day-block bootstrap of a split's Sharpe, paired with a reference policy on the same days.

    returns / ref_returns: per-step net returns on the same split. Resamples whole days with
    replacement (the same days for both), so the spread reflects how much the score depends on
    which days the split happened to contain. Returns the 5 / 95 % Sharpe bounds and the share
    of resamples where the policy beats the reference.
    """
    spd = steps_per_day(step_gap)
    ppy = 365 * spd
    n = len(returns) // spd
    a = np.asarray(returns, float)[: n * spd].reshape(n, spd)
    b = np.asarray(ref_returns, float)[: n * spd].reshape(n, spd)
    rng = np.random.default_rng(seed)
    sa, sb = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        d = rng.integers(0, n, n)
        sa[i], sb[i] = sharpe(a[d].ravel(), ppy), sharpe(b[d].ravel(), ppy)
    return {"sharpe_p5": float(np.percentile(sa, 5)), "sharpe_p95": float(np.percentile(sa, 95)),
            "beats_ref_pct": 100 * float(np.mean(sa > sb))}
