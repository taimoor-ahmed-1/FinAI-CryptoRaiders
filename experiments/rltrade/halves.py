"""
Experiment 23c: Double DQN on the REAL factors with a decision every 60 bars, checkpoint chosen honestly.

E23a showed that a decision every 60 bars lets Double DQN learn a planted signal at IC 0.1 (the honest factors sit
near 0.1-0.13), but training is unstable: the validation Sharpe of successive checkpoints swings widely, so "the final
policy" and "the best validation checkpoint" are both poor estimates. Test and stress_bear stay locked, so the
validation period is cut in two (walkforward.windows: V1 = first half, V2 = second half):

    train   the agent learns here (cross-fitted factors, so train forecasts are out-of-sample)
    V1      every checkpoint is scored here (exact pass); the best one with >= min_trades trades is kept
    V2      the kept checkpoint is judged here, once - it never influenced training or the choice

The same rule as E16 applies on top: if the kept checkpoint does not beat both cash and buy and hold on V1, the
run trades as cash ("stay-flat fallback"). Both the raw checkpoint and the fallback result are reported. The
greedy policy is also scored on train (chunked) at every check, for the learning curve.

E23d adds a second way to choose (`select_by`). V1 and V2 are opposite markets (buy and hold Sharpe -1.43 on V1,
+3.68 on V2), and in E23c the best-V1-Sharpe checkpoint was always the mostly-short one, which then lost on V2: the
choice picked up the market's direction instead of skill. "skill" ranks checkpoints by what they earn on V1 beyond
their average exposure, net of costs:

    skill % = timing % - costs %       (diagnose.attribution: timing = gross - mean level x market move)

so the direction of V1 cannot decide the choice. Both choices are always tracked and reported (`selections`);
`selected` is the one named by `select_by`. The stay-flat rule follows the choice: "sharpe" stays flat unless V1
Sharpe beats max(0, buy and hold), "skill" stays flat unless V1 skill is positive.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch as th

from . import seeding
from .agents import make_agent
from .baselines import buy_and_hold_policy
from .diagnose import attribution, daily_timing
from .env import EnvConfig, Market, TradingEnv, run_policy
from .learnability import CURVE_KEYS, score_policy, with_market_return
from .metrics import compute_metrics
from .walkforward import windows

SPLITS = ("train", "v1", "v2", "val")
SELECT_BY = ("sharpe", "skill")


def write_halves(prepared_dir, factor_dir, out_dir):
    """Writes market_<s>.npz and factors_<s>.npy for train, v1, v2 and val (val = v1 + v2) into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = {s: dict(np.load(Path(prepared_dir) / f"market_{s}.npz")) for s in ("train", "val")}
    f = {s: np.load(Path(factor_dir) / f"factors_{s}.npy").astype(np.float32) for s in ("train", "val")}
    n_val = len(f["val"])
    w = windows(len(f["train"]), n_val)
    cuts = {"train": ("train", 0, len(f["train"])), "v1": w["V1"], "v2": w["V2"], "val": ("val", 0, n_val)}
    for name, (s, lo, hi) in cuts.items():
        np.savez(out / f"market_{name}.npz",
                 **{k: (a[lo:hi] if a.ndim and len(a) == len(f[s]) else a) for k, a in m[s].items()})
        np.save(out / f"factors_{name}.npy", f[s][lo:hi])
    info = {name: {"split": s, "rows": [int(lo), int(hi)]} for name, (s, lo, hi) in cuts.items()}
    (out / "halves.json").write_text(json.dumps({"factor_dir": str(factor_dir), "cuts": info}, indent=1))
    return out, info


def reference(data_dir, cfg, device):
    """Exact buy-and-hold metrics on every split."""
    return {s: compute_metrics(run_policy(buy_and_hold_policy, Market(data_dir, data_dir, s, cfg, device), cfg, chunks=1),
                               cfg.periods_per_year) for s in SPLITS}


def _exact(policy, market, cfg):
    bt = run_policy(policy, market, cfg, chunks=1)
    m = compute_metrics(bt, cfg.periods_per_year)
    return {k: m[k] for k in CURVE_KEYS}, bt


def _timing(bt, market, cfg):
    """timing % and skill % (= timing - costs) of one exact pass."""
    a = attribution(with_market_return(bt, market, cfg), cfg.step_gap)
    return {"timing_pct": a["timing_pct"], "skill_pct": a["timing_pct"] - a["costs_pct"]}


def _score(row, by):
    return row["v1_sharpe"] if by == "sharpe" else row["v1_skill_pct"]


def train_select(name, data_dir, out_root, seed, agent_hp, env_over, total_steps, eval_every, device="auto",
                 min_trades=5, train_chunks=32, verbose=True, select_by="sharpe"):
    """Trains Double DQN on train; at step 0 and every `eval_every` steps scores the greedy policy on train
    (chunked) and exactly on V1 and V2. Among checkpoints with at least `min_trades` V1 trades it keeps the best
    one by V1 Sharpe and the best one by V1 skill, and reports both on V2; `selected` is the one named by
    `select_by`. Resumes: a run whose result.json exists is loaded."""
    assert select_by in SELECT_BY, select_by
    out = Path(out_root) / "runs" / name
    if (out / "result.json").exists():
        if verbose:
            print(f"[skip] {name} (already done)")
        return json.loads((out / "result.json").read_text())
    t0 = time.time()
    dev = seeding.pick_device(device)
    seeding.set_seed(seed)
    cfg = EnvConfig(state="A", **env_over)
    mk = {s: Market(data_dir, data_dir, s, cfg, dev) for s in SPLITS}
    bh = reference(data_dir, cfg, dev)
    agent = make_agent("ddqn", cfg.state_dim, 3, dev, seed, **agent_hp)
    env = TradingEnv(mk["train"], cfg, num_envs=agent.cfg.num_envs, mode="train", seed=seed)
    curve = []
    best = {by: {"score": -np.inf, "steps": None, "state": None} for by in SELECT_BY}

    def callback(ag, steps):
        pol = lambda o: ag.act(o, deterministic=True)        # noqa: E731
        tr, _ = score_policy(pol, mk["train"], cfg, train_chunks)
        v1, bt1 = _exact(pol, mk["v1"], cfg)
        v2, bt2 = _exact(pol, mk["v2"], cfg)
        v1.update(_timing(bt1, mk["v1"], cfg))
        v2.update(_timing(bt2, mk["v2"], cfg))
        row = {"steps": steps, **{f"train_{k}": v for k, v in tr.items()},
               **{f"v1_{k}": v for k, v in v1.items()}, **{f"v2_{k}": v for k, v in v2.items()}}
        eligible = v1["num_trades"] >= min_trades
        row["eligible"] = bool(eligible)
        state = None
        for by in SELECT_BY:
            if eligible and _score(row, by) > best[by]["score"]:
                if state is None:
                    state = {k: {n: t.detach().cpu().clone() for n, t in v.items()} for k, v in ag.state_dict().items()}
                best[by].update(score=_score(row, by), steps=steps, state=state)
        curve.append(row)
        if verbose:
            print(f"  {name} | {steps:>9,} | train sharpe {tr['sharpe']:+.2f} | V1 sharpe {v1['sharpe']:+.2f} "
                  f"skill {v1['skill_pct']:+.1f}% trades {v1['num_trades']} | V2 sharpe {v2['sharpe']:+.2f} "
                  f"skill {v2['skill_pct']:+.1f}% trades {v2['num_trades']}"
                  + ("" if eligible else f" (< {min_trades} V1 trades)"))
        return False

    callback(agent, 0)
    agent.learn(env, total_steps, callback=callback, callback_every=eval_every)
    if curve[-1]["steps"] < total_steps - eval_every // 2:
        callback(agent, total_steps)
    final_pol = lambda o: agent.act(o, deterministic=True)   # noqa: E731
    final = {s: _exact(final_pol, mk[s], cfg)[0] for s in ("v1", "v2", "val")}
    selections = {}
    for by in SELECT_BY:
        b = best[by]
        if b["state"] is None:
            selections[by] = None
            continue
        agent.load_state_dict(b["state"])
        pol = lambda o: agent.act(o, deterministic=True)     # noqa: E731
        sel = {}
        for s in ("v1", "v2"):
            m, bt = _exact(pol, mk[s], cfg)
            trm = with_market_return(bt, mk[s], cfg)
            a = attribution(trm, cfg.step_gap)
            sel[s] = {**m, "skill_pct": a["timing_pct"] - a["costs_pct"], **{f"attr_{k}": v for k, v in a.items()},
                      **{f"daily_{k}": v for k, v in daily_timing(trm, cfg.step_gap).items()}}
        bar = max(0.0, bh["v1"]["sharpe"]) if by == "sharpe" else 0.0
        score = sel["v1"]["sharpe"] if by == "sharpe" else sel["v1"]["skill_pct"]
        selections[by] = {"steps": b["steps"], "by": by, "fallback": bool(score <= bar), "v1_bar": bar, **sel}
        if by == select_by:
            out.mkdir(parents=True, exist_ok=True)
            th.save(b["state"], out / "model_selected.pt")
    selected = selections[select_by]
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(agent.history))
    res = {"name": name, "seed": seed, "data_dir": str(data_dir), "env": {"state": "A", **env_over},
           "agent_hp": {k: list(v) if isinstance(v, tuple) else v for k, v in agent_hp.items()},
           "total_steps": total_steps, "min_trades": min_trades, "select_by": select_by, "curve": curve,
           "final": final, "selected": selected, "selections": selections,
           "buy_and_hold": {s: {k: bh[s][k] for k in CURVE_KEYS} for s in SPLITS},
           "minutes": round((time.time() - t0) / 60, 2), "commit": seeding.run_info(seed)["commit"]}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    if verbose:
        s = selected
        txt = (f"selected ({select_by}) step {s['steps']:,}: V1 sharpe {s['v1']['sharpe']:+.2f} "
               f"skill {s['v1']['skill_pct']:+.1f}% -> V2 sharpe {s['v2']['sharpe']:+.2f} "
               f"ret {s['v2']['cum_return_pct']:+.1f}%" + (" (fallback: cash)" if s["fallback"] else "")) if s else \
              f"no checkpoint with >= {min_trades} V1 trades"
        print(f"[done] {name} | {txt} | final V2 sharpe {final['v2']['sharpe']:+.2f} | {res['minutes']} min")
    return res
