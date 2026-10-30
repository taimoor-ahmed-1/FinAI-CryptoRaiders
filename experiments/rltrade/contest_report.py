"""
Report for experiment 09 (contest week, news signals) -> <contest_root>/report/

    python -m rltrade.contest_report <contest_root>

    01_contest_week_price_and_news.png   price with the 70/15/15 split, the two news signals
    02_ablation_realistic_costs.png      A-D at 0.12 % per trade, bar = mean of 10 seeds, dots = seeds
    03_ablation_low_costs.png            A and D at 0.02 % per trade
    04_equity_curves.png                 mean test equity per state, both cost levels
    tables/*.csv, README.md              numbers, Welch tests, eligibility and trading activity

Same figure style and helpers as compile_results (the main thesis results).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import contest
from .compile_results import (INK2, SLOTS, STATE_NAMES, STYLE, _curve, _perf_figure, _plt, _save,
                              _seed_table, _split_spans, _welch)
from .results import df_markdown

PAIRS = (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"))
LABELS = {k: v.split("  ")[0] + " " + v.split("  ")[1] for k, v in STATE_NAMES.items()}


def _market(prepared):
    frames = []
    for s in ("train", "val", "test"):
        with np.load(Path(prepared) / f"market_{s}.npz") as z:
            frames.append(pd.DataFrame({"t": pd.to_datetime(z["time_ns"], unit="ns", utc=True), "split": s,
                                        "price": z["midpoint"], "sentiment": z["sentiment_score"],
                                        "risk": z["risk_score"]}))
    return pd.concat(frames, ignore_index=True)


def fig_data(prepared, path):
    m = _market(prepared)
    spans = {s: (g.t.iloc[0], g.t.iloc[-1]) for s, g in m.groupby("split", sort=False)}
    plt = _plt()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
        axes[0].plot(m.t, m.price, color=INK2, lw=1)
        axes[0].set_ylabel("BTC price (USD)")
        axes[0].set_title("01  Contest week: BTC price, the 70/15/15 split and the news signals the agent sees", loc="left")
        for ax, col, c, name in ((axes[1], "sentiment", SLOTS[0], "news sentiment"), (axes[2], "risk", SLOTS[1], "news risk")):
            ax.step(m.t, m[col], where="post", color=c, lw=1.2)
            ax.set_ylim(0.8, 5.2)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_ylabel(f"{name}\n(1-5)")
        for ax in axes:
            _split_spans(ax, spans)
        for s, (a, b) in spans.items():
            axes[0].annotate(s, (a + (b - a) / 2, 0.97), xycoords=("data", "axes fraction"), ha="center", va="top",
                             fontsize=8.5, color=INK2)
        fig.text(0.01, -0.01, "Score of the most recent news article available before each 1-minute bar ends "
                 "(rebuilt from publication times; held until the next article).", fontsize=8.5, color=INK2, ha="left")
        fig.tight_layout()
        _save(fig, path)


def _tests(sub, pairs):
    rows = []
    for a, b in pairs:
        x, y = sub[sub.state == a], sub[sub.state == b]
        if len(x) > 1 and len(y) > 1:
            for metric in ("test_sharpe", "test_cum_return_pct"):
                t, p = _welch(x[metric].values, y[metric].values)
                rows.append({"comparison": f"{a} vs {b}", "metric": metric,
                             "difference": x[metric].mean() - y[metric].mean(), "welch_t": t, "p_value": p})
    return pd.DataFrame(rows, columns=["comparison", "metric", "difference", "welch_t", "p_value"])


def report(root):
    P = contest.contest_layout(root)
    out = P["report"]
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df = contest.load_all(root)
    rl = df[df.agent == contest.AGENT]
    base = df[df.agent.isin(["buy_and_hold", "cash"])]
    real, low = rl[rl.tag == ""], rl[rl.tag == "lowcost"]
    tables, lines = {}, []

    fig_data(P["prepared"], out / "01_contest_week_price_and_news.png")
    for name, sub, states, b, num in (("realistic", real, "ABCD", base[base.tag == ""], "02"),
                                      ("low", low, "AD", base[base.tag == "lowcost"], "03")):
        if sub.empty:
            continue
        tab = _seed_table(sub, "state", list(states), LABELS)
        elig = sub.groupby("state").selection_eligible.apply(lambda s: int((s.astype(str) == "True").sum()))
        tab["seeds_with_eligible_policy"] = [int(elig.get(s, 0)) for s in states if (sub.state == s).any()]
        # median and before-cost return: a few high-turnover checkpoints dominate the means at 0.12 %
        g = sub.assign(gross=sub.test_cum_return_pct + sub.test_costs_pct).groupby("state")
        present = [s for s in states if (sub.state == s).any()]
        tab["test_return_%_median"] = g.test_cum_return_pct.median().reindex(present).to_numpy()
        tab["test_return_before_costs_%_mean"] = g.gross.mean().reindex(present).to_numpy()
        tab["test_costs_%_mean"] = g.test_costs_pct.mean().reindex(present).to_numpy()
        tests = _tests(sub, [p for p in PAIRS if set(p) <= set(states)])
        tables[f"{num}_ablation_{name}_costs"], tables[f"{num}_tests_{name}_costs"] = tab, tests
        sh = tests[tests.metric == "test_sharpe"]
        ptxt = ";  ".join(f"{r.comparison}: p = {r.p_value:.2f}" for r in sh.itertuples())
        cost = "0.12 %" if name == "realistic" else "0.02 %"
        _perf_figure(sub, "state", list(states), LABELS,
                     f"{num}  Contest week, {cost} per trade: do the news signals help? Double DQN, bar = mean of "
                     f"{sub.seed.nunique()} seeds, dots = seeds",
                     out / f"{num}_ablation_{name}_costs.png", b,
                     footnote=(f"Welch t-tests on test Sharpe:  {ptxt}.  " if ptxt else "")
                              + "Test window is ~20 hours, so seeds share one market path and p-values are indicative only.")
        lines += [f"## {num} Ablation, {cost} per trade\n", df_markdown(tab, 3), "", df_markdown(tests, 4), ""]

    plt = _plt()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, sub, tag, cost in ((axes[0], real, "", "0.12 %"), (axes[1], low, "lowcost", "0.02 %")):
            for i, s in enumerate("ABCD"):
                g = sub[sub.state == s]
                if g.empty:
                    continue
                x, y = _curve(P["runs"] / g.group.iloc[0], g.run_id)
                if x is not None:
                    ax.plot(x, y.mean(0), color=SLOTS[i], lw=1.8, label=LABELS[s])
            for a, ls, lab in (("buy_and_hold", "-", "Buy & hold"), ("cash", ":", "Cash")):
                g = base[(base.agent == a) & (base.tag == tag)]
                if len(g):
                    x, y = _curve(P["runs"] / "baselines", g.run_id)
                    if x is not None:
                        ax.plot(x, y.mean(0), color=INK2, lw=1.3, ls=ls, label=lab)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{cost} per trade")
            ax.tick_params(axis="x", rotation=30)
        axes[0].set_ylabel("cumulative test return (%)")
        handles = {}
        for ax in axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        fig.suptitle("04  Test-period equity, mean over seeds per state", fontweight="bold", fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.0))
        _save(fig, out / "04_equity_curves.png")

    tables["05_baselines"] = base[["agent", "tag", "test_cum_return_pct", "test_sharpe", "test_num_trades"]]
    tables["06_all_runs"] = df
    for k, t in tables.items():
        t.to_csv(out / "tables" / f"{k}.csv", index=False)
    head = ["# Experiment 09 - contest week (news signals)\n",
            "FinAI Contest 2025 Task 1 order book, 2021-04-07 -> 04-13, with the organisers' DeepSeek scores of 279 "
            "Bitcoin news articles, rebuilt causally from publication times (`rltrade/contest.py`). Same pipeline as the "
            "thesis: Alpha101 -> the GRU selected in 04 (applied unchanged) -> Double DQN with the hyperparameters "
            "tuned in 06; checkpoint chosen on validation (>= 5 trades), test evaluated once.\n",
            "**Caveat:** 5.7 days in total; validation and test are ~20 hours each. Seeds measure training noise on one "
            "short test window, not market variation, so the p-values are indicative only. Annualised Sharpe ratios "
            "from 20 hours are large in magnitude; compare them between states, not with the main thesis.\n"]
    lines += ["## Baselines (test)\n", df_markdown(tables["05_baselines"], 3), ""]
    text = "\n".join(head + lines)
    (out / "README.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(report(sys.argv[1]))
