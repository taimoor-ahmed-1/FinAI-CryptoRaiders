"""
Experiment 12 - interval-valued, multi-scale news signals and interval-constrained trading, after
Tian, Yang, Wei & Wang (2026), "Interval-valued forecasting and trading strategy for Bitcoin: a multi-scale
framework with multimodal data", Pacific-Basin Finance Journal 103352.

    python -m rltrade.contest_intervals build    --root <contest_root> --news news_train.csv
    python -m rltrade.contest_intervals precheck --root <contest_root> --news news_train.csv
    python -m rltrade.contest_intervals run      --root <contest_root> --hp best_hp_mt5.json --groups E,F
    python -m rltrade.contest_intervals report   --root <contest_root>

What is taken from the paper, and how it is adapted to one week of 1-minute bars
    intervals    the paper summarises a week of article sentiment by its 25th and 75th percentiles
                 (robust to single outlier articles). Here, for every bar, [q25, q75] of the DeepSeek
                 sentiment and risk scores of the articles available in a trailing window; the agent sees
                 the interval's centre (level) and width (how much the news disagrees).
    scales       the paper splits every series into low / medium / high-frequency parts (NA-MEMD over the
                 whole sample - that looks ahead, so it is not used). Here two causal windows: 6 h (~10
                 articles, the fast part) and 24 h (~40 articles, the slow part).
    attention    the paper uses Google Trends. Here news intensity from the same feed:
                 log((articles in 6 h + 1) / (articles in 24 h / 4 + 1)), > 0 when news is busier than usual.
    trading      the paper trades only when its point forecast lies inside its forecast interval, i.e. when
                 uncertainty is acceptable. Here (state F) the environment lets the agent open or enlarge a
                 position only while the 6 h sentiment interval is no wider than its train-split median;
                 reducing or closing is always allowed; with fewer than MIN_ARTICLES articles in the window
                 there is no interval and the gate is open.
    scorer       the paper uses VADER; here the organisers' DeepSeek scores (VADER rated 20 % of these articles
                 exactly 0 and gives no risk score - see the E12 VADER pre-check in EXPERIMENTS.md).

Timing and articles exactly as in experiment 09 (a bar sees only articles available before it ends;
availability = the later of date_time and the Yahoo URL time), restricted to the articles that mention
bitcoin / crypto (as in E11; the paper used a crypto-only news source). Each feature is centred on its train
mean and divided by the train 99th percentile of |x - mean|, clipped to [-1, 1]; no interval -> 0 (the train
mean). All settings are fixed before any run; nothing is tuned on validation or test.

The E9 market files are copied to <root>/prepared_e12 with the new columns added; the E9 factors are reused.
States E (= A + the 9 interval columns) and F (= E + gate) are compared with E9's A (price only) and D (point
scores), same DDQN, hyperparameters, seeds, episodes, budget and costs.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import contest
from .env import INTERVAL_COLS

EXPERIMENT = "12_contest_intervals"
WINDOWS = {"6h": 6 * 3600, "24h": 24 * 3600}
MIN_ARTICLES = 3
GROUPS = ("E", "F", "E_lowcost", "F_lowcost")
RELEVANT = re.compile(r"bitcoin|\bbtc\b|crypto|blockchain|coinbase|ether|\beth\b|miner|mining|digital asset"
                      r"|digital currenc|altcoin|defi|stablecoin|satoshi|grayscale|microstrategy|dogecoin|binance")
SPLITS = ("train", "val", "test")


def layout(root):
    root = Path(root)
    return {**contest.contest_layout(root), "prepared_e12": root / "prepared_e12", "runs_e12": root / "runs_e12",
            "report_e12": root / "report_e12"}


def relevant(news):
    """Title or the first 600 characters of the text mention bitcoin / crypto (same rule as starter_kit E11)."""
    text = (news["title"].fillna("") + " " + news["article_text"].fillna("").str[:600]).str.lower()
    return text.str.contains(RELEVANT).to_numpy()


def interval_features(bar_start_ns, news, train_rows, bar_seconds=60):
    """Per-bar raw features (DataFrame) from articles available strictly before each bar ends."""
    news = news[relevant(news)].copy()
    news["available"] = contest.news_availability(news)
    news = news.sort_values("available", kind="stable")
    a = np.asarray(pd.to_datetime(news["available"], utc=True).map(pd.Timestamp.timestamp), dtype=float)
    end = np.asarray(bar_start_ns, dtype=np.int64) / 1e9 + bar_seconds
    hi = np.searchsorted(a, end, side="left")                     # articles with available < bar end
    out, counts = {}, {}
    for wn, w in WINDOWS.items():
        lo = np.searchsorted(a, end - w, side="left")             # ... and available >= bar end - window
        counts[wn] = hi - lo
        for col, short in (("sentiment_score", "sent"), ("risk_score", "risk")):
            x = news[col].to_numpy(float)
            cache, centre, width = {}, np.full(len(end), np.nan), np.full(len(end), np.nan)
            for i, (l, h) in enumerate(zip(lo, hi)):
                if h - l < MIN_ARTICLES:
                    continue
                if (l, h) not in cache:
                    q25, q75 = np.quantile(x[l:h], [0.25, 0.75])
                    cache[(l, h)] = ((q25 + q75) / 2, q75 - q25)
                centre[i], width[i] = cache[(l, h)]
            out[f"{short}_centre_{wn}"], out[f"{short}_width_{wn}"] = centre, width
    out["news_intensity"] = np.log((counts["6h"] + 1) / (counts["24h"] / 4 + 1))
    raw = pd.DataFrame(out)[list(INTERVAL_COLS)]
    # gate: shut while the 6 h sentiment interval is wider than its train median; open without an interval
    w6 = raw["sent_width_6h"].to_numpy()
    med = float(np.nanmedian(w6[:train_rows]))
    gate = np.where(np.isnan(w6), 1.0, (w6 <= med).astype(float))
    return raw, gate, {"gate_width_threshold": med, "articles": int(len(news)), "counts": counts}


def scale(raw, train_rows):
    """Centre on the train mean, divide by the train 99th percentile of |x - mean|, clip to [-1, 1], NaN -> 0."""
    out, stats = {}, {}
    for c in raw.columns:
        v = raw[c].to_numpy(float)
        ref = v[:train_rows][~np.isnan(v[:train_rows])]
        m = float(ref.mean())
        s = float(np.quantile(np.abs(ref - m), 0.99)) or 1.0
        out[c] = np.nan_to_num(np.clip((v - m) / s, -1, 1), nan=0.0).astype(np.float32)
        stats[c] = {"train_mean": m, "scale": s, "train_coverage": round(float(np.mean(~np.isnan(v[:train_rows]))), 3)}
    return out, stats


def _load_markets(prepared):
    ms = {s: dict(np.load(Path(prepared) / f"market_{s}.npz")) for s in SPLITS}
    return ms, {s: len(ms[s]["midpoint"]) for s in SPLITS}


def build(root, news_csv):
    P = layout(root)
    ms, n = _load_markets(P["prepared"])
    t_ns = np.concatenate([ms[s]["time_ns"] for s in SPLITS])
    raw, gate, info = interval_features(t_ns, pd.read_csv(news_csv), n["train"])
    feats, stats = scale(raw, n["train"])
    P["prepared_e12"].mkdir(parents=True, exist_ok=True)
    meta = {"windows_s": WINDOWS, "min_articles": MIN_ARTICLES, "articles": info["articles"],
            "gate_width_threshold": info["gate_width_threshold"], "features": stats, "splits": {}}
    r0 = 0
    for s in SPLITS:
        r1 = r0 + n[s]
        new = {c: feats[c][r0:r1] for c in INTERVAL_COLS}
        new["gate_open"] = gate[r0:r1].astype(np.float32)
        np.savez(P["prepared_e12"] / f"market_{s}.npz", **ms[s], **new)
        meta["splits"][s] = {"bars": n[s], "gate_open_share": round(float(gate[r0:r1].mean()), 3),
                             "interval_6h_share": round(float((~raw["sent_width_6h"].iloc[r0:r1].isna()).mean()), 3),
                             "changes_sent_centre_6h": int((np.diff(new["sent_centre_6h"]) != 0).sum())}
        r0 = r1
    shutil.copy(P["prepared"] / "meta.json", P["prepared_e12"] / "meta_e9.json")
    (P["prepared_e12"] / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return meta


def precheck(root, news_csv, n_check=300, n_shift=200, seed=0):
    """Causality (recompute sampled bars from the raw articles) and information (Spearman IC with the future
    return, train and val only, p-value from circular shifts of the feature)."""
    P = layout(root)
    ms = {s: dict(np.load(P["prepared_e12"] / f"market_{s}.npz")) for s in SPLITS}
    meta = json.loads((P["prepared_e12"] / "meta.json").read_text())
    news = pd.read_csv(news_csv)
    news = news[relevant(news)].copy()
    news["available"] = pd.to_datetime(contest.news_availability(news), utc=True)
    rng = np.random.default_rng(seed)
    t_ns = np.concatenate([ms[s]["time_ns"] for s in SPLITS])
    allf = {c: np.concatenate([ms[s][c] for s in SPLITS]) for c in INTERVAL_COLS + ("gate_open",)}
    bad = 0
    for i in rng.choice(len(t_ns), n_check, replace=False):
        end = pd.Timestamp(int(t_ns[i]), unit="ns", tz="UTC") + pd.Timedelta(seconds=60)
        exp = {}
        for wn, w in WINDOWS.items():
            sub = news[(news["available"] < end) & (news["available"] >= end - pd.Timedelta(seconds=w))]
            exp[wn] = len(sub)
            for col, short in (("sentiment_score", "sent"), ("risk_score", "risk")):
                for kind in ("centre", "width"):
                    c = f"{short}_{kind}_{wn}"
                    if len(sub) < MIN_ARTICLES:
                        v = 0.0
                    else:
                        q25, q75 = np.quantile(sub[col].to_numpy(float), [0.25, 0.75])
                        x = (q25 + q75) / 2 if kind == "centre" else q75 - q25
                        st = meta["features"][c]
                        v = float(np.clip((x - st["train_mean"]) / st["scale"], -1, 1))
                    bad += not np.isclose(allf[c][i], v, atol=1e-5)
        st = meta["features"]["news_intensity"]
        v = np.clip((np.log((exp["6h"] + 1) / (exp["24h"] / 4 + 1)) - st["train_mean"]) / st["scale"], -1, 1)
        bad += not np.isclose(allf["news_intensity"][i], v, atol=1e-5)
    print(f"causality check: {n_check} bars x 9 features, {bad} mismatches")

    rows = []
    for s in ("train", "val"):
        lp = np.log(ms[s]["midpoint"])
        for c in INTERVAL_COLS + ("gate_open",):
            f = ms[s][c]
            for hn, h in (("15 min", 15), ("1 h", 60), ("4 h", 240)):
                fut = lp[h:] - lp[:-h]
                x = pd.Series(f[:-h])
                if x.nunique() < 2:
                    rows.append(dict(split=s, feature=c, horizon=hn, ic=np.nan, p_shift=np.nan))
                    continue
                y = pd.Series(fut)
                ic = x.corr(y, method="spearman")
                null = [pd.Series(np.roll(f[:-h], k)).corr(y, method="spearman")
                        for k in rng.integers(len(y) // 10, len(y) - len(y) // 10, n_shift)]
                p = (np.sum(np.abs(null) >= abs(ic)) + 1) / (n_shift + 1)
                rows.append(dict(split=s, feature=c, horizon=hn, ic=round(ic, 3), p_shift=round(p, 3)))
    t = pd.DataFrame(rows)
    out = P["report_e12"]
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "precheck_ic.csv", index=False)
    piv = t.assign(v=t.ic.map("{:+.3f}".format) + " (" + t.p_shift.map("{:.2f}".format) + ")") \
        .pivot_table(index=["feature", "split"], columns="horizon", values="v", aggfunc="first", sort=False)
    print(piv[["15 min", "1 h", "4 h"]].to_string())
    return t, bad


def run_groups(root, hp_path, groups=GROUPS, seeds=contest.SEEDS, device="auto", threads=0):
    from .runner import run
    if threads:
        import torch
        torch.set_num_threads(threads)
    P = layout(root)
    hp = json.loads(Path(hp_path).read_text())[contest.AGENT]
    for g in groups:
        for spec in contest.group_specs(g, hp["agent_hp"], hp["env"], seeds, experiment=EXPERIMENT):
            run(spec, P["prepared_e12"], P["factors"], P["runs_e12"] / g, device=device)


def report(root):
    """E9's A and D against E12's E and F at both cost levels: mean (median) over seeds + Welch tests."""
    from .compile_results import _welch
    from .results import load_results
    P = layout(root)
    e9 = contest.load_all(root)
    e12 = pd.concat([load_results(d).assign(group=d.name) for d in sorted(P["runs_e12"].iterdir())
                     if (d / "results.csv").exists()], ignore_index=True)
    e12["costs"] = np.where(e12.tag == "lowcost", "low (0.02 %)", "realistic (0.12 %)")
    df = pd.concat([e9[(e9.agent == "ddqn") & e9.state.isin(["A", "D"])], e12[e12.agent == "ddqn"],
                    e9[e9.agent != "ddqn"]], ignore_index=True)
    df["test_return_before_costs_pct"] = df.test_cum_return_pct + df.test_costs_pct
    df["name"] = np.where(df.agent == "ddqn", df.state, df.agent)        # baselines are logged with state D
    rows, tests = [], []
    for costs, sub in df.groupby("costs", sort=False):
        for name, g in sub.groupby("name", sort=False):
            rows.append({"costs": costs, "state": name, "seeds": len(g),
                         **{f"{m}_{k}": getattr(g[f"test_{m}"], k)() for m in
                            ("cum_return_pct", "sharpe", "max_drawdown_pct") for k in ("mean", "median")},
                         "return_before_costs_pct_mean": g.test_return_before_costs_pct.mean(),
                         "trades_median": g.test_num_trades.median(),
                         "seeds_profitable": int((g.test_cum_return_pct > 0).sum()),
                         "val_sharpe_mean": g.val_sharpe.mean()})
        dd = sub[sub.agent == "ddqn"]
        for a, b in (("E", "A"), ("F", "A"), ("E", "D"), ("F", "E")):
            x, y = dd[dd.state == a], dd[dd.state == b]
            if len(x) > 1 and len(y) > 1:
                for m in ("test_sharpe", "test_cum_return_pct"):
                    t, p = _welch(x[m].values, y[m].values)
                    tests.append({"costs": costs, "comparison": f"{a} vs {b}", "metric": m,
                                  "difference": x[m].mean() - y[m].mean(), "welch_t": t, "p_value": p})
    out = P["report_e12"]
    out.mkdir(parents=True, exist_ok=True)
    summary, tests = pd.DataFrame(rows), pd.DataFrame(tests)
    summary.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "tests.csv", index=False)
    df.to_csv(out / "all_runs.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))
    return summary, tests


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rltrade.contest_intervals")
    ap.add_argument("cmd", choices=("build", "precheck", "run", "report"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--news")
    ap.add_argument("--hp")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--seeds", default=",".join(map(str, contest.SEEDS)))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.root, a.news)
    elif a.cmd == "precheck":
        precheck(a.root, a.news)
    elif a.cmd == "run":
        run_groups(a.root, a.hp, a.groups.split(","), [int(x) for x in a.seeds.split(",")], a.device, a.threads)
    else:
        report(a.root)


if __name__ == "__main__":
    main()
