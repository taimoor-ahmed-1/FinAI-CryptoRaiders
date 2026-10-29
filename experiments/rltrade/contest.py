"""
Contest-week dataset: FinAI Contest 2025 Task 1 LOB data + the organisers' news scores.

    python -m rltrade.contest BTC_1sec.csv news_train.csv contest_1min_news.csv

Inputs (never committed - the repo is public)
    BTC_1sec.csv     contest limit-order-book file, 1-second rows, 2021-04-07 11:32 -> 2021-04-13 05:00 UTC
    news_train.csv   HF dataset SecureFinAI-Lab/FinRL_BTC_news_signals: 279 Bitcoin news articles of
                     the same week, each scored by DeepSeek (1-5 sentiment, 1-5 risk), with its
                     publication time `date_time` (UTC)

Output: one CSV of 1-minute bars in the schema `prepare.prepare` reads, with INTRADAY signals.

MARKET BARS - built exactly like the thesis's Binance bars (build_market_data.py), so the GRU factor
model trained on 2024-26 sees the same kind of inputs:
    midpoint   (high + low) / 2 of the 1-second midpoints in the minute
    spread     high - low of those midpoints (an intra-bar range, as in the thesis data)
    buys/sells summed over the minute
    bids_distance_3 / asks_distance_3   (low - mid) / mid and (high - mid) / mid  (the thesis overload)
    bids_notional_3 / asks_notional_3   visible depth: the 15 contest levels summed, at the minute's
                                        last second (the thesis used cumulative depth within 4 %)
    system_time is the bar START (floor of the minute).

NEWS SIGNALS - rebuilt causally from publication times instead of using the organisers' merged file:
    available_at  = the later of `date_time` and the time stamped in the article's Yahoo URL
                    (they agree for 236 of 240 Yahoo articles; the later one is used when they differ)
    bar value     = mean score of the most recently available article(s) (ties averaged, as the
                    organisers did), using only articles available BEFORE THE BAR ENDS - the same
                    moment the bar's price becomes known; the agent trades one bar later (exec_lag)
    before the first article: the train-period mean
The scores are already time-aligned, so `prepare` must be called with signal_lag="none".

EXPERIMENT 09 - do the news signals improve the RL agent? (ablation A-D, as in 05)
    agent     DDQN with the hyperparameters tuned in 06 (best_hp_mt5.json) - not re-tuned: validation
              is ~20 hours, far too little to tune on
    factors   the GRU selected in 04, applied unchanged (rnn.train.export_factors)
    split     the contest's chronological 70/15/15 (train ~4 days, val ~20 h, test ~20 h)
    episodes  12 hours (the thesis's 3-day episodes do not fit a 4-day train split)
    budget    300k transitions, validation every 10k, exact single-pass validation
    costs     realistic 0.10 % fee + 0.02 % slippage (as in the thesis), states A-D
              + a low-cost check at 0.02 % per trade (maker level), states A and D
    seeds     10 per configuration; buy & hold and cash as references at both cost levels
Each group writes to its own `<runs>/<group>/` folder (own results.csv), so groups can run in
parallel processes without two writers sharing one results file.

    python -m rltrade.contest run --root <contest_root> --hp best_hp_mt5.json --groups A,B
    python -m rltrade.contest report --root <contest_root>
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import split_bounds

LEVELS = range(15)
SCORES = ("sentiment_score", "risk_score")
_YAHOO_TIME = re.compile(r"-(\d{2})(\d{2})(\d{2})\d{3}\.html")


def build_bars(sec):
    """1-second contest rows (DataFrame) -> 1-minute bars."""
    t = pd.to_datetime(sec["system_time"], utc=True, format="ISO8601")
    g = sec.assign(bar=t.dt.floor("min")).groupby("bar", sort=True)
    hi, lo = g["midpoint"].max(), g["midpoint"].min()
    mid = (hi + lo) / 2.0
    last = g.tail(1).set_index("bar")
    bars = pd.DataFrame({
        "system_time": hi.index.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "midpoint": mid.to_numpy(),
        "spread": (hi - lo).to_numpy(),
        "buys": g["buys"].sum().to_numpy(),
        "sells": g["sells"].sum().to_numpy(),
        "bids_distance_3": ((lo - mid) / mid).to_numpy(),
        "asks_distance_3": ((hi - mid) / mid).to_numpy(),
        "bids_notional_3": last[[f"bids_notional_{i}" for i in LEVELS]].sum(axis=1).to_numpy(),
        "asks_notional_3": last[[f"asks_notional_{i}" for i in LEVELS]].sum(axis=1).to_numpy(),
    })
    return bars


def news_availability(news):
    """Time from which each article may be used: max(date_time, Yahoo URL time)."""
    t = pd.to_datetime(news["date_time"], utc=True, format="ISO8601")
    avail = t.copy()
    for i, (ti, url) in enumerate(zip(t, news["url"].fillna(""))):
        m = _YAHOO_TIME.search(url) if "finance.yahoo.com" in url else None
        if not m:
            continue
        h, mi, s = map(int, m.groups())
        u = ti.normalize() + pd.Timedelta(hours=h, minutes=mi, seconds=s)
        if u < ti - pd.Timedelta(hours=12):              # URL clock is past midnight: next calendar day
            u += pd.Timedelta(days=1)
        avail.iloc[i] = max(ti, u)
    return avail


def news_signals(bar_start, news, bar_seconds=60):
    """Per-bar scores from articles available strictly before each bar's end."""
    av = news_availability(news)
    per_t = (news.assign(available_at=av).groupby("available_at")[list(SCORES)].mean()
             .sort_index().reset_index())
    ends = pd.DataFrame({"end": pd.to_datetime(bar_start, utc=True, format="ISO8601")
                         + pd.Timedelta(seconds=bar_seconds)})
    per_t["available_at"] = per_t["available_at"].astype(ends["end"].dtype)
    m = pd.merge_asof(ends, per_t, left_on="end", right_on="available_at",
                      direction="backward", allow_exact_matches=False)
    return m[list(SCORES)].reset_index(drop=True)


def build_dataset(sec_csv, news_csv, out_csv=None, train_ratio=0.70, val_ratio=0.15):
    cols = ["system_time", "midpoint", "buys", "sells"] + \
           [f"{s}_notional_{i}" for s in ("bids", "asks") for i in LEVELS]
    sec = pd.read_csv(sec_csv, usecols=cols)
    bars = build_bars(sec)
    news = pd.read_csv(news_csv)
    sig = news_signals(bars["system_time"], news)
    train_end = split_bounds(len(bars), train_ratio, val_ratio)["train"][1]
    for c in SCORES:
        bars[c] = sig[c].fillna(sig[c].iloc[:train_end].mean()).to_numpy()
    if out_csv is not None:
        bars.to_csv(out_csv, index=False)
    return bars


# ------------------------------------------------------------------ experiment 09
EXPERIMENT = "09_contest_news"
AGENT = "ddqn"
SEEDS = tuple(range(10))
TOTAL_STEPS, EVAL_EVERY = 300_000, 10_000
ENV = {"episode_bars": 720, "start_skip": 400}           # 12-hour episodes; skip the factor warm-up
LOW_COST = {"fee_rate": 0.0001, "slippage": 0.0001}      # 0.02 % per trade
GROUPS = ("A", "B", "C", "D", "A_lowcost", "D_lowcost", "baselines")


def group_specs(group, agent_hp, env_over, seeds=SEEDS):
    """RunSpecs of one group: '<state>' or '<state>_lowcost' (DDQN x seeds), or 'baselines'."""
    from .env import EnvConfig
    from .runner import RunSpec
    if group == "baselines":
        return [RunSpec(EXPERIMENT, b, seed=0, tag=tag, env=EnvConfig(state="D", **ENV, **costs))
                for tag, costs in (("", {}), ("lowcost", LOW_COST)) for b in ("buy_and_hold", "cash")]
    state, _, low = group.partition("_")
    costs, tag = (LOW_COST, "lowcost") if low else ({}, "")
    env = EnvConfig(state=state, **ENV, **env_over, **costs)
    return [RunSpec(EXPERIMENT, AGENT, seed=s, env=env, agent_hp=agent_hp, tag=tag, total_steps=TOTAL_STEPS,
                    eval_every=EVAL_EVERY, val_chunks=1) for s in seeds]


def contest_layout(root):
    root = Path(root)
    return {"data": root / "data" / "contest_1min_news.csv", "prepared": root / "prepared",
            "factors": root / "factors", "runs": root / "runs", "report": root / "report"}


def run_groups(root, hp_path, groups=GROUPS, seeds=SEEDS, device="auto", threads=0):
    from .runner import run
    if threads:
        import torch
        torch.set_num_threads(threads)
    P = contest_layout(root)
    hp = json.loads(Path(hp_path).read_text())[AGENT]
    for g in groups:
        for spec in group_specs(g, hp["agent_hp"], hp["env"], seeds):
            run(spec, P["prepared"], P["factors"], P["runs"] / g, device=device)


def load_all(root):
    """results.csv rows of every group folder, with a `costs` column."""
    from .results import load_results
    runs = contest_layout(root)["runs"]
    df = pd.concat([load_results(d).assign(group=d.name) for d in sorted(runs.iterdir())
                    if (d / "results.csv").exists()], ignore_index=True)
    df["costs"] = np.where(df.tag == "lowcost", "low (0.02 %)", "realistic (0.12 %)")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rltrade.contest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="BTC_1sec.csv + news_train.csv -> 1-minute dataset")
    b.add_argument("sec_csv"), b.add_argument("news_csv"), b.add_argument("out_csv")
    r = sub.add_parser("run", help="train / evaluate experiment groups")
    r.add_argument("--root", required=True), r.add_argument("--hp", required=True)
    r.add_argument("--groups", default=",".join(GROUPS))
    r.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    r.add_argument("--device", default="auto"), r.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        bars = build_dataset(a.sec_csv, a.news_csv, a.out_csv)
        print(f"{len(bars):,} bars {bars.system_time.iloc[0]} -> {bars.system_time.iloc[-1]} | "
              + " | ".join(f"{c}: {(bars[c].diff() != 0).sum() - 1} changes, mean {bars[c].mean():.2f}"
                           for c in SCORES) + f" -> {Path(a.out_csv).resolve()}")
    elif a.cmd == "run":
        run_groups(a.root, a.hp, a.groups.split(","), [int(x) for x in a.seeds.split(",")], a.device, a.threads)


if __name__ == "__main__":
    main()
