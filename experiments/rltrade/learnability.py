"""
Experiment 21: is the Double DQN agent learning anything? Train and validation only.

"Not learning" can mean three different things, and each needs a different fix:

1. it cannot learn at all: even on its own training data the greedy policy does no better than at the start
   (the reward, state, decision interval or costs make the problem unlearnable);
2. it learns the training data but nothing that carries over (overfitting / a changing market);
3. it learns, but not from the factors (a direction bet that ignores the inputs).

Four tests separate them, all with the E16 Double DQN (state A, E16 hyperparameters, no entry gate):

A  real factors      train/val curves: the greedy policy is scored on BOTH train and validation every
                      `eval_every` steps (and once before any training), so in-sample learning is visible.
B  shuffled factors   the same factor values with whole days moved to random other days (train and validation
                      separately): the factors keep their distribution and smoothness but carry no information.
                      If A and B look the same, the agent is not using the factors.
C  planted signal     factor f5 (the 300-bar factor) replaced by a noisy copy of the FUTURE 300-bar return with a
                      chosen correlation (IC) - a deliberate leak, used only as a positive control. The weakest IC
                      the agent turns into profit is the setup's learnability bound; the honest factors sit near 0.1.
D  decision vs signal the agent's preference for "go long" over "go short" from a flat position, Q(increase) -
                      Q(decrease), at every bar, correlated with each factor and with the forward return.

Each variant also gets the E19 benchmark: the best threshold rule on train, traded on validation, so a failure of
the agent can be compared with what a simple rule extracts from the same inputs.

Nothing here selects a checkpoint for reporting a trading result: the curves show every checkpoint.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from scipy import stats

from . import seeding
from .agents import make_agent
from .baselines import buy_and_hold_policy
from .diagnose import attribution, daily_timing
from .env import EnvConfig, Market, TradingEnv, run_policy
from .factor_rule import OFFSETS, forward_returns, ic
from .metrics import compute_metrics

PLANT_FACTOR = 5                     # f5 forecasts the price 300 bars ahead
PLANT_H = OFFSETS[PLANT_FACTOR]      # 300
DAY = 1440
CURVE_KEYS = ("sharpe", "cum_return_pct", "max_drawdown_pct", "num_trades", "costs_pct", "long_pct", "short_pct",
              "exposure_pct")


# ---------------------------------------------------------------- factor variants (tests B and C)
def shuffle_days(factors, seed, block=DAY):
    """Whole `block`-row pieces of the factor matrix in a random order (a trailing partial block stays last).
    Every row keeps its 8 values, so the distribution and the within-day smoothness are unchanged, but each
    day's factors now sit next to another day's prices."""
    f = np.asarray(factors)
    k = len(f) // block
    order = np.random.default_rng(seed).permutation(k)
    pieces = [f[i * block:(i + 1) * block] for i in order] + [f[k * block:]]
    return np.concatenate(pieces).astype(f.dtype)


def planted_signal(price, rho, seed, h=PLANT_H, lag=1):
    """rho * z + sqrt(1 - rho^2) * noise, with z the standardised log return from the execution price of bar t
    (t + lag) to h bars later, and noise an independent random walk's h-bar increment (so the signal is as smooth
    as z). Rows without a full future window get z = 0. Unit variance, correlation with z about rho."""
    fwd = np.log1p(forward_returns(price, h, lag))
    ok = np.isfinite(fwd)
    z = np.zeros(len(price))
    z[ok] = (fwd[ok] - fwd[ok].mean()) / fwd[ok].std()
    e = np.random.default_rng(seed).standard_normal(len(price) + h)
    c = np.concatenate([[0.0], np.cumsum(e)])
    noise = c[h:h + len(price)] - c[:len(price)]
    noise = (noise - noise.mean()) / noise.std()
    s = rho * z + np.sqrt(1.0 - rho ** 2) * noise
    return (s - s.mean()) / s.std()


def write_variant(prepared_dir, factor_dir, out_dir, kind, rho=0.0, seed=0, splits=("train", "val")):
    """Writes factors_{split}.npy for a variant ("shuffled" or "planted") plus variant.json with the realised IC
    of factor f5 against the 300-bar forward return. The planted column is put on the scale of the original
    f5 on train (same mean and standard deviation)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = {s: np.load(Path(factor_dir) / f"factors_{s}.npy").astype(np.float32) for s in splits}
    mu, sd = float(src["train"][:, PLANT_FACTOR].mean()), float(src["train"][:, PLANT_FACTOR].std())
    info = {"kind": kind, "rho": rho, "seed": seed, "factor": PLANT_FACTOR, "horizon": PLANT_H, "ic": {}}
    for k, s in enumerate(splits):
        f = src[s].copy()
        price = np.load(Path(prepared_dir) / f"market_{s}.npz")["midpoint"].astype(np.float64)
        if kind == "shuffled":
            f = shuffle_days(f, seed + 1000 * k)
        elif kind == "planted":
            f[:, PLANT_FACTOR] = mu + sd * planted_signal(price, rho, seed + 1000 * k)
        else:
            raise ValueError(kind)
        np.save(out / f"factors_{s}.npy", f.astype(np.float32))
        info["ic"][s] = ic(f[:, PLANT_FACTOR], forward_returns(price, PLANT_H), PLANT_H)[0]
    (out / "variant.json").write_text(json.dumps(info, indent=1))
    return out, info


# ---------------------------------------------------------------- training with train + validation curves (A-C)
def _gross_pct(bt):
    return 100.0 * float(np.sum(np.asarray(bt["returns"]) + np.asarray(bt["cost"])))


def score_policy(policy, market, cfg, chunks):
    bt = run_policy(policy, market, cfg, chunks=chunks)
    m = compute_metrics(bt, cfg.periods_per_year)
    return {**{k: m[k] for k in CURVE_KEYS}, "gross_pct": _gross_pct(bt)}, bt


def with_market_return(bt, market, cfg):
    """Adds `mret` (the market's return over each decision step) so diagnose.attribution can split the result."""
    p = market.price.cpu().numpy()
    b, g, lag = bt["bar"], cfg.step_gap, cfg.exec_lag
    return {**bt, "mret": p[b + g + lag] / p[b + lag] - 1.0}


def train_run(name, prepared_dir, factor_dir, out_root, seed, agent_hp, env_over, total_steps, eval_every,
              device="auto", train_chunks=32, val_chunks=16, verbose=True):
    """Trains Double DQN on train and scores the greedy policy on train AND validation at step 0 and every
    `eval_every` steps; the final policy also gets one exact pass on each split. Resumes: a run whose
    result.json exists is loaded, not retrained."""
    out = Path(out_root) / "runs" / name
    if (out / "result.json").exists():
        if verbose:
            print(f"[skip] {name} (already done)")
        return json.loads((out / "result.json").read_text())
    t0 = time.time()
    dev = seeding.pick_device(device)
    seeding.set_seed(seed)
    cfg = EnvConfig(state="A", **env_over)
    mk = {s: Market(prepared_dir, factor_dir, s, cfg, dev) for s in ("train", "val")}
    agent = make_agent("ddqn", cfg.state_dim, 3, dev, seed, **agent_hp)
    env = TradingEnv(mk["train"], cfg, num_envs=agent.cfg.num_envs, mode="train", seed=seed)
    chunks = {"train": train_chunks, "val": val_chunks}
    curve = []

    def callback(ag, steps):
        pol = lambda o: ag.act(o, deterministic=True)        # noqa: E731
        row = {"steps": steps}
        for s in ("train", "val"):
            m, _ = score_policy(pol, mk[s], cfg, chunks[s])
            row.update({f"{s}_{k}": v for k, v in m.items()})
        curve.append(row)
        if verbose:
            print(f"  {name} | {steps:>9,} steps | train sharpe {row['train_sharpe']:+.2f} ret {row['train_cum_return_pct']:+.1f}% "
                  f"trades {row['train_num_trades']} | val sharpe {row['val_sharpe']:+.2f} ret {row['val_cum_return_pct']:+.1f}% "
                  f"trades {row['val_num_trades']}")
        return False

    callback(agent, 0)                                     # the untrained policy: where learning starts from
    agent.learn(env, total_steps, callback=callback, callback_every=eval_every)
    if curve[-1]["steps"] < total_steps - eval_every // 2:
        callback(agent, total_steps)
    pol = lambda o: agent.act(o, deterministic=True)       # noqa: E731
    final = {}
    for s in ("train", "val"):
        m, bt = score_policy(pol, mk[s], cfg, 1)
        tr = with_market_return(bt, mk[s], cfg)
        final[s] = {**m, **{f"attr_{k}": v for k, v in attribution(tr, cfg.step_gap).items()},
                    **{f"daily_{k}": v for k, v in daily_timing(tr, cfg.step_gap).items()}}
    out.mkdir(parents=True, exist_ok=True)
    th.save(agent.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(agent.history))
    res = {"name": name, "seed": seed, "factor_dir": str(factor_dir), "env": {"state": "A", **env_over},
           "agent_hp": {k: list(v) if isinstance(v, tuple) else v for k, v in agent_hp.items()},
           "total_steps": total_steps, "curve": curve, "final": final, "minutes": round((time.time() - t0) / 60, 2),
           "commit": seeding.run_info(seed)["commit"]}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    if verbose:
        f = final
        print(f"[done] {name} | final exact: train sharpe {f['train']['sharpe']:+.2f} ret {f['train']['cum_return_pct']:+.1f}% | "
              f"val sharpe {f['val']['sharpe']:+.2f} ret {f['val']['cum_return_pct']:+.1f}% | {res['minutes']} min")
    return res


def load_agent(run_dir, device="auto"):
    res = json.loads((Path(run_dir) / "result.json").read_text())
    cfg = EnvConfig(**res["env"])
    hp = {k: tuple(v) if isinstance(v, list) else v for k, v in res["agent_hp"].items()}
    dev = seeding.pick_device(device)
    agent = make_agent("ddqn", cfg.state_dim, 3, dev, res["seed"], **hp)
    agent.load_state_dict(th.load(Path(run_dir) / "model.pt", map_location=dev))
    return agent, cfg


def reference_scores(market, cfg, chunks):
    """Buy and hold with the same chunking as the curves (cash is 0 by definition)."""
    m, _ = score_policy(buy_and_hold_policy, market, cfg, chunks)
    return m


# ---------------------------------------------------------------- test D: what drives the decision
@th.no_grad()
def q_preference(agent, market, batch=65536):
    """Q(increase) - Q(decrease) at every bar, from a flat position that has held nothing (state A):
    > 0 = the agent would go long, < 0 = short. Also returns the greedy action from flat (0/1/2)."""
    feat = market.feat[:, :8].to(agent.device)
    prefs, acts = [], []
    for i in range(0, len(feat), batch):
        f = feat[i:i + batch]
        obs = th.cat([th.zeros(len(f), 2, device=agent.device), f], dim=1)
        q = agent.q(obs)
        prefs.append((q[:, 2] - q[:, 0]).float().cpu())
        acts.append(q.argmax(-1).cpu())
    return th.cat(prefs).numpy().astype(np.float64), th.cat(acts).numpy()


def preference_table(pref, actions, factors, price, horizons=(60, PLANT_H, 900), every=60):
    """Spearman correlation of the preference with each factor (sampled every `every` bars), its IC against the
    forward return at each horizon (non-overlapping samples, factor_rule.ic), and the action mix from flat."""
    row = {}
    s = slice(None, None, every)
    for j in range(factors.shape[1]):
        f = np.asarray(factors[s, j], np.float64)
        row[f"corr_f{j}"] = float(stats.spearmanr(pref[s], f)[0]) if np.std(pref[s]) > 0 and np.std(f) > 0 else float("nan")
    for h in horizons:
        rho, p, _ = ic(pref, forward_returns(price, h), h)
        row[f"ic_{h}"], row[f"ic_{h}_p"] = rho, p
    row.update({"from_flat_long_pct": 100 * float(np.mean(actions == 2)),
                "from_flat_short_pct": 100 * float(np.mean(actions == 0)),
                "from_flat_stay_pct": 100 * float(np.mean(actions == 1))})
    return row


__all__ = ["PLANT_FACTOR", "PLANT_H", "shuffle_days", "planted_signal", "write_variant", "train_run", "load_agent",
           "reference_scores", "q_preference", "preference_table", "score_policy", "with_market_return"]
