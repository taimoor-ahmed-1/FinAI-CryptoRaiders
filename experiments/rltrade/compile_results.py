"""Supervisor-ready results: numbered figures, one table per figure, and a README walkthrough.

    from rltrade import compile_results
    compile_results.compile_all(L)          # -> <root>/thesis_results/

Rebuilt from scratch each time, from `runs/results.csv`, the runs' test backtests,
`prepared/` and `rnn/tuning/final_runs.csv`. Only the current, leak-free experiments are
shown (06_agents_mt5, 05_ablation_ddqn_mt5, 07_rewards, lagged signals); superseded runs
and the same-day (look-ahead) leak-check runs stay on Drive but are left out here.

    thesis_results/
      README.md                one section per figure: what it shows, the table, how to read it
      01_...png .. 11_...png   numbered in reading order
      tables/01_...csv ..      the numbers behind each figure
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import results as R

OUT_NAME = "thesis_results"
EXP_AGENTS, EXP_ABLATION, EXP_REWARDS = "06_agents_mt5", "05_ablation_ddqn_mt5", "07_rewards"
RL_AGENTS = ["ppo", "a2c", "dqn", "ddqn", "d3qn"]
AGENT_NAMES = {"ppo": "PPO", "a2c": "A2C", "dqn": "DQN", "ddqn": "Double DQN", "d3qn": "Dueling DDQN"}
STATE_NAMES = {"A": "A  price only", "B": "B  + sentiment", "C": "C  + risk", "D": "D  + sentiment + risk"}
REWARD_NAMES = {"net_asset_change": "Net-asset change", "diff_sharpe": "Differential Sharpe",
                "drawdown_penalized": "Drawdown-penalised"}
CONFIG_STATE = {"A": "price only (A)", "B": "+ sentiment (B)", "C": "+ risk (C)"}

# reference palette (light mode): categorical slots in fixed order, text and surface tokens
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
STYLE = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
         "axes.titleweight": "bold", "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
         "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
         "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5, "figure.dpi": 110}


# ----------------------------------------------------------------------------- data selection
def current_runs(df):
    """The runs the thesis reports: lagged (causal) signals, current eligibility rule."""
    keep = df.experiment.isin([EXP_AGENTS, EXP_ABLATION, EXP_REWARDS]) & (df.signal_source == "lagged")
    return df[keep].copy()


def _seed_table(sub, label_col, order, names):
    rows = []
    for key in order:
        s = sub[sub[label_col] == key]
        if s.empty:
            continue
        rows.append({label_col: names.get(key, key), "seeds": len(s),
                     "val_sharpe_mean": s.val_sharpe.mean(),
                     "test_return_%_mean": s.test_cum_return_pct.mean(), "test_return_%_std": s.test_cum_return_pct.std(),
                     "test_return_%_min": s.test_cum_return_pct.min(), "test_return_%_max": s.test_cum_return_pct.max(),
                     "test_sharpe_mean": s.test_sharpe.mean(), "test_sharpe_std": s.test_sharpe.std(),
                     "max_drawdown_%_mean": s.test_max_drawdown_pct.mean(),
                     "trades_median": s.test_num_trades.median(),
                     "seeds_profitable": int((s.test_cum_return_pct > 0).sum())})
    return pd.DataFrame(rows)


def _welch(x, y):
    from scipy import stats
    t = stats.ttest_ind(x, y, equal_var=False)
    return float(t.statistic), float(t.pvalue)


def static_policy_table(df):
    """Seeds whose test policy is (almost) static: 1 trade = enter once and hold."""
    d = df[~df.agent.isin(["buy_and_hold", "cash", "random"])].copy()
    d["trades"] = pd.to_numeric(d.test_num_trades, errors="coerce")
    g = d.groupby(["experiment", "agent", "state", "signal_source", "reward"], dropna=False)
    return g.agg(seeds=("trades", "size"), never_trade=("trades", lambda s: int((s == 0).sum())),
                 one_trade=("trades", lambda s: int((s == 1).sum())), le_5_trades=("trades", lambda s: int((s <= 5).sum())),
                 median_trades=("trades", "median")).reset_index()


def final_model(df, min_trades=5):
    """Same rule as notebook 08: best mean validation Sharpe over all reported configurations."""
    cand = df[~df.agent.isin(["buy_and_hold", "random", "cash"]) & (df.signal_source == "lagged")]
    cand = cand[pd.to_numeric(cand.select_min_trades, errors="coerce") >= min_trades]
    if cand.empty:
        return None, None
    best = cand.groupby(R.GROUP, dropna=False).val_sharpe.mean().idxmax()
    return dict(zip(R.GROUP, best)), cand.set_index(R.GROUP).loc[[best]].reset_index()


# ----------------------------------------------------------------------------- plotting helpers
def _plt():
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    _plt().close(fig)


def _bars_with_seeds(ax, labels, groups, ylabel, fmt="{:+.1f}", highlight=None, note=None):
    """Bar = mean over seeds, dots = individual seeds (so the spread is visible)."""
    x = np.arange(len(labels))
    means = [np.mean(g) if len(g) else np.nan for g in groups]
    colors = [SLOTS[0] if highlight is None or i == highlight else "#b7d3f6" for i in range(len(labels))]
    ax.bar(x, means, width=0.6, color=colors, zorder=2)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        jit = rng.uniform(-0.13, 0.13, len(g))
        ax.scatter(x[i] + jit, g, s=26, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.axhline(0, color=INK2, lw=0.9, zorder=1)
    ax.set_xticks(x, [f"{lab}\nmean {fmt.format(m)}" if not np.isnan(m) else lab for lab, m in zip(labels, means)])
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if note:
        ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")


def _ref_line(ax, y, label):
    ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color=INK2)


def _curve(runs_dir, run_ids):
    t, a = R._curves(runs_dir, list(run_ids))
    if a is None:
        return None, None
    x = R.as_datetime(np.concatenate([[t[0]], t]))[:a.shape[1]] if t is not None else np.arange(a.shape[1])
    return x, (a - 1.0) * 100


def _split_spans(ax, splits):
    for i, (name, (a, b)) in enumerate(splits.items()):
        ax.axvspan(a, b, color=SLOTS[[0, 3, 1][i % 3]], alpha=0.09, lw=0)


# ----------------------------------------------------------------------------- the figures
def fig_data(L, out, tables):
    """01 price and splits, 02 daily LLM signals."""
    plt = _plt()
    frames = []
    for s in ("train", "val", "test"):
        f = Path(L.prepared) / f"market_{s}.npz"
        if not f.exists():
            return []
        z = np.load(f)
        cols = {"price": z["midpoint"]}
        for c in ("sentiment_score", "risk_score", "tweet_volume"):
            k = f"sameday__{c}"
            if k in z.files:
                cols[c] = z[k]
        d = pd.DataFrame(cols, index=R.as_datetime(z["time_ns"]))
        d["split"] = s
        frames.append(d)
    bars = pd.concat(frames)
    day = bars.resample("1D").agg({**{c: "last" if c == "price" else "mean" for c in bars.columns if c != "split"},
                                   "split": "first"}).dropna(subset=["price"])
    spans = {s: (g.index.min(), g.index.max()) for s, g in bars.groupby("split", sort=False)}
    names = {"train": "Train (70 %)", "val": "Validation (15 %)", "test": "Test (15 %)"}

    rows = []
    for s, g in bars.groupby("split", sort=False):
        rows.append({"split": s, "from": str(g.index.min())[:16], "to": str(g.index.max())[:16], "days": round(len(g) / 1440, 1),
                     "one_minute_bars": len(g), "btc_start": round(g.price.iloc[0]), "btc_end": round(g.price.iloc[-1]),
                     "buy_and_hold_%": round((g.price.iloc[-1] / g.price.iloc[0] - 1) * 100, 1)})
    tables["01_data_splits"] = pd.DataFrame(rows)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        _split_spans(ax, spans)
        ax.plot(day.index, day.price, color=INK, lw=1.4)
        for i, (s, (a, b)) in enumerate(spans.items()):
            r = tables["01_data_splits"].iloc[i]
            ax.text(a + (b - a) / 2, 0.04, f"{names[s]}\nBTC {r['buy_and_hold_%']:+.0f} %", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_ylabel("BTC price (USD)")
        ax.set_title("01  Bitcoin price and the chronological train / validation / test split")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        _save(fig, out / "01_data_btc_price_and_splits.png")

    sig = [c for c in ("sentiment_score", "risk_score", "tweet_volume") if c in day]
    if sig:
        labels = {"sentiment_score": "Sentiment\n(1 bearish - 5 bullish)", "risk_score": "Risk\n(1 calm - 5 high)",
                  "tweet_volume": "Tweets\nper day"}
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(len(sig), 1, figsize=(10, 2.1 * len(sig)), sharex=True)
            axes = np.atleast_1d(axes)
            for ax, c, col in zip(axes, sig, [SLOTS[0], SLOTS[1], MUTED]):
                _split_spans(ax, spans)
                ax.plot(day.index, day[c], color=col, lw=0.6, alpha=0.45)
                ax.plot(day.index, day[c].rolling(30, min_periods=10).mean(), color=col, lw=1.8)
                ax.set_ylabel(labels[c], fontsize=8.5)
            axes[0].set_title("02  Daily LLM-scored tweet signals (thin = daily mean, thick = 30-day average)")
            _save(fig, out / "02_data_daily_llm_signals.png")
        rows = []
        for s, g in day.groupby("split", sort=False):
            rows.append({"split": s, **{f"{c}_mean": round(g[c].mean(), 3) for c in sig}, **{f"{c}_std": round(g[c].std(), 3) for c in sig}})
        tables["02_data_signals"] = pd.DataFrame(rows)
    return ["01", "02"] if sig else ["01"]


def fig_rnn(L, out, tables):
    f = Path(L.rnn_tuning) / "final_runs.csv"
    if not f.exists():
        return []
    plt = _plt()
    fr = pd.read_csv(f)
    order = fr.groupby("arch").val_mse.mean().sort_values().index.tolist()
    tab = fr.groupby("arch").agg(seeds=("seed", "size"), val_mse_mean=("val_mse", "mean"), val_mse_std=("val_mse", "std"),
                                 val_ic_mean=("val_ic", "mean"), test_ic_mean=("test_ic", "mean")).loc[order].reset_index()
    tab["selected"] = tab.arch == order[0]
    tables["03_factor_model"] = tab
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        _bars_with_seeds(axes[0], order, [fr[fr.arch == a].val_mse.values for a in order], "validation MSE (lower is better)",
                         fmt="{:.4f}", highlight=0)
        lo = fr.val_mse.min()
        axes[0].set_ylim(lo - (fr.val_mse.max() - lo) * 3 - 1e-4, fr.val_mse.max() + (fr.val_mse.max() - lo) * 1.5 + 1e-4)
        axes[0].set_title("Validation error (selection metric)")
        _bars_with_seeds(axes[1], order, [fr[fr.arch == a].test_ic.values for a in order],
                         "test information coefficient", fmt="{:.3f}", highlight=0)
        axes[1].set_title("Test IC (correlation of factor with future return)")
        fig.suptitle(f"03  Factor-model architectures - {order[0]} selected (dark bar); dots = seeds",
                     fontweight="bold", fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        _save(fig, out / "03_factor_model_architectures.png")
    return ["03"]


def _perf_figure(sub, key, order, names, title, path, base=None, ref=True, footnote=None):
    plt = _plt()
    order = [k for k in order if (sub[key] == k).any()]
    labels = [names.get(k, k) for k in order]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        _bars_with_seeds(axes[0], labels, [sub[sub[key] == k].test_cum_return_pct.values for k in order],
                         "test return, net of costs (%)", fmt="{:+.1f}%")
        _bars_with_seeds(axes[1], labels, [sub[sub[key] == k].test_sharpe.values for k in order],
                         "test Sharpe ratio", fmt="{:+.2f}")
        axes[0].set_title("Return over the test period")
        axes[1].set_title("Risk-adjusted return")
        if base is not None and ref:
            bh = base[base.agent == "buy_and_hold"]
            if len(bh):
                _ref_line(axes[0], bh.test_cum_return_pct.iloc[0], f"buy & hold {bh.test_cum_return_pct.iloc[0]:+.1f}%")
                _ref_line(axes[1], bh.test_sharpe.iloc[0], f"buy & hold {bh.test_sharpe.iloc[0]:+.2f}")
        fig.suptitle(title, fontweight="bold", fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        if footnote:
            fig.text(0.01, -0.03, footnote, fontsize=8.5, color=INK2, ha="left", va="top")
        _save(fig, path)


def fig_agents(df, runs_dir, out, tables):
    sub = df[(df.experiment == EXP_AGENTS) & df.agent.isin(RL_AGENTS)]
    base = df[(df.experiment == EXP_AGENTS) & df.agent.isin(["buy_and_hold", "cash"])]
    if sub.empty:
        return []
    tab = _seed_table(sub, "agent", RL_AGENTS, AGENT_NAMES)
    if "selection_eligible" in sub:
        elig = sub.groupby("agent").selection_eligible.apply(lambda s: int((s.astype(str) == "True").sum()))
        tab["seeds_with_eligible_policy"] = [int(elig.get(a, 0)) for a in RL_AGENTS if (sub.agent == a).any()]
    tables["04_agents"] = tab
    _perf_figure(sub, "agent", RL_AGENTS, AGENT_NAMES,
                 "04  Five RL algorithms (state D, net-asset reward) - bar = mean of 5 seeds, dots = seeds; cash = 0",
                 out / "04_agents_test_performance.png", base)

    plt = _plt()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for i, a in enumerate(RL_AGENTS):
            x, y = _curve(runs_dir, sub[sub.agent == a].run_id)
            if x is None:
                continue
            ax.plot(x, y.mean(0), color=SLOTS[i], lw=1.8, label=AGENT_NAMES[a])
        for a, ls, lab in (("buy_and_hold", "-", "Buy & hold"), ("cash", ":", "Cash (never trade)")):
            x, y = _curve(runs_dir, base[base.agent == a].run_id)
            if x is not None:
                ax.plot(x, y.mean(0), color=INK2, lw=1.4, ls=ls, label=lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel("cumulative return (%)")
        ax.set_title("05  Test-period equity: mean over 5 seeds per algorithm")
        ax.legend(ncol=4, loc="lower left")
        _save(fig, out / "05_agents_equity_curves.png")
    return ["04", "05"]


def fig_ablation(df, out, tables, welch_out):
    abl = df[((df.experiment == EXP_ABLATION) & (df.agent == "ddqn"))
             | ((df.experiment == EXP_AGENTS) & (df.agent == "ddqn") & (df.state == "D"))]
    if abl.empty or abl.state.nunique() < 2:
        return []
    tables["06_ablation"] = _seed_table(abl, "state", list("ABCD"), STATE_NAMES)
    rows = []
    for a, b in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B")):
        x, y = abl[abl.state == a].test_sharpe.values, abl[abl.state == b].test_sharpe.values
        if len(x) > 1 and len(y) > 1:
            t, p = _welch(x, y)
            rows.append({"comparison": f"{a} vs {b}", "sharpe_difference": x.mean() - y.mean(), "welch_t": t, "p_value": p})
    welch_out.extend(rows)
    tables["06_ablation_tests"] = pd.DataFrame(rows)
    ptxt = ";  ".join(f"{r['comparison']}: p = {r['p_value']:.2f}" for r in rows)
    _perf_figure(abl, "state", list("ABCD"), {k: v.split("  ")[0] + " " + v.split("  ")[1] for k, v in STATE_NAMES.items()},
                 "06  Do the LLM signals help? The same tuned Double DQN with four versions of its input",
                 out / "06_ablation_llm_signals.png", df[(df.experiment == EXP_AGENTS)],
                 footnote=f"Welch t-tests on test Sharpe (5 seeds each):  {ptxt}.  p < 0.05 would indicate a real difference.")
    return ["06"]


def fig_rewards(df, out, tables):
    rw = df[(df.experiment == EXP_REWARDS)]
    if rw.empty:
        return []
    order = list(REWARD_NAMES)
    tables["07_rewards"] = _seed_table(rw, "reward", order, REWARD_NAMES)
    _perf_figure(rw, "reward", order, REWARD_NAMES,
                 "07  Reward functions (tuned Double DQN, state D) - bar = mean of 5 seeds, dots = seeds",
                 out / "07_reward_functions.png", df[(df.experiment == EXP_AGENTS)])
    return ["07"]


def fig_final(df, runs_dir, out, tables):
    cfg, runs = final_model(df)
    if cfg is None:
        return [], None
    plt = _plt()
    runs = runs.sort_values("seed")
    tables["08_final_model"] = runs[["seed", "val_sharpe", "test_sharpe", "test_cum_return_pct", "test_max_drawdown_pct",
                                     "test_num_trades"]].rename(columns={"test_cum_return_pct": "test_return_%",
                                                                         "test_max_drawdown_pct": "max_drawdown_%",
                                                                         "test_num_trades": "test_trades"})
    name = f"{AGENT_NAMES.get(cfg['agent'], cfg['agent'])}, state {cfg['state']}, {REWARD_NAMES.get(cfg['reward'], cfg['reward'])} reward"
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        y = np.arange(len(runs))
        for yi, (_, r) in zip(y, runs.iterrows()):
            ax.plot([r.val_sharpe, r.test_sharpe], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(runs.val_sharpe, y, s=70, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3, label="validation")
        ax.scatter(runs.test_sharpe, y, s=70, color=SLOTS[1], edgecolor=SURFACE, linewidth=2, zorder=3, label="test")
        for yi, (_, r) in zip(y, runs.iterrows()):
            ntr = int(r.test_num_trades)
            ax.annotate(f"{r.test_cum_return_pct:+.0f}% ({ntr} trade{'s' if ntr != 1 else ''})", (r.test_sharpe, yi),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.set_yticks(y, [f"seed {int(s)}" for s in runs.seed])
        ax.set_ylim(-0.6, len(runs) - 0.3)
        ax.set_xlabel("Sharpe ratio")
        ax.set_title(f"08  Final model ({name}): validation vs test, per seed")
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)
        _save(fig, out / "08_final_model_validation_vs_test.png")

        fig, ax = plt.subplots(figsize=(11, 4.2))
        for i, (_, r) in enumerate(runs.iterrows()):
            x, yv = _curve(runs_dir, [r.run_id])
            if x is not None:
                ax.plot(x, yv[0], color=SLOTS[i % len(SLOTS)], lw=1.5, label=f"seed {int(r.seed)}")
        bh = df[(df.experiment == EXP_AGENTS) & (df.agent == "buy_and_hold")]
        x, yv = _curve(runs_dir, bh.run_id)
        if x is not None:
            ax.plot(x, yv.mean(0), color=INK2, lw=1.4, ls=(0, (4, 3)), label="Buy & hold")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel("cumulative return (%)")
        ax.set_title(f"09  Final model on the test period, each seed separately")
        ax.legend(ncol=6, loc="upper left")
        _save(fig, out / "09_final_model_equity_per_seed.png")
    return ["08", "09"], (cfg, name)


def _configs(df):
    """(label, section, rows) for every reported configuration, each once."""
    out = []
    for a in RL_AGENTS:
        out.append((f"{AGENT_NAMES[a]} (state D)", "Algorithms (state D)", df[(df.experiment == EXP_AGENTS) & (df.agent == a)]))
    for s in "ABC":
        out.append((f"Double DQN, {CONFIG_STATE[s]}", "LLM-signal ablation",
                    df[(df.experiment == EXP_ABLATION) & (df.agent == "ddqn") & (df.state == s)]))
    for rw in ("diff_sharpe", "drawdown_penalized"):
        out.append((f"Double DQN, {REWARD_NAMES[rw].lower()} reward", "Rewards", df[(df.experiment == EXP_REWARDS) & (df.reward == rw)]))
    return [c for c in out if len(c[2])]


def fig_activity(df, out, tables):
    plt = _plt()
    cfgs = _configs(df)
    if not cfgs:
        return []
    rows = [{"configuration": lab, "seeds": len(s), "median_trades": s.test_num_trades.median(),
             "seeds_0_or_1_trade": int((s.test_num_trades <= 1).sum()), "seeds_le_5_trades": int((s.test_num_trades <= 5).sum())}
            for lab, _, s in cfgs]
    tables["10_trading_activity"] = pd.DataFrame(rows)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(cfgs) + 1.2))
        ax.axvspan(-0.5, 5.5, color=SLOTS[1], alpha=0.10, lw=0)
        ax.text(2.5, len(cfgs) - 0.55, "near-static\n(5 or fewer trades)", ha="center", va="bottom", fontsize=8, color=INK2)
        rng = np.random.default_rng(1)
        for i, (lab, sec, s) in enumerate(cfgs):
            v = s.test_num_trades.values.astype(float)
            ax.scatter(np.maximum(v, 0) + rng.uniform(-0.15, 0.15, len(v)) * (v < 6), i + rng.uniform(-0.12, 0.12, len(v)),
                       s=30, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xscale("symlog", linthresh=10)
        ticks = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if t <= max(10, df.test_num_trades.max() * 1.5)]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        for i in range(1, len(cfgs)):
            if cfgs[i][1] != cfgs[i - 1][1]:
                ax.axhline(i - 0.5, color=GRID, lw=1)
        ax.set_yticks(range(len(cfgs)), [c[0] for c in cfgs])
        ax.set_ylim(len(cfgs) - 0.4, -0.6)
        ax.set_xlim(-0.5, max(10, df.test_num_trades.max() * 1.5))
        ax.set_xlabel("trades in the test period (one dot per seed; scale compressed above 10)")
        ax.set_title("10  How actively does each trained policy trade?")
        ax.grid(axis="y", visible=False)
        _save(fig, out / "10_trading_activity.png")
    return ["10"]


def fig_overview(df, out, tables):
    plt = _plt()
    cfgs = _configs(df)
    if not cfgs:
        return []
    bh = df[(df.experiment == EXP_AGENTS) & (df.agent == "buy_and_hold")]
    rows = [{"configuration": lab, "group": sec, "test_sharpe_mean": s.test_sharpe.mean(), "test_sharpe_min": s.test_sharpe.min(),
             "test_sharpe_max": s.test_sharpe.max(), "test_return_%_mean": s.test_cum_return_pct.mean(),
             "val_sharpe_mean": s.val_sharpe.mean()} for lab, sec, s in cfgs]
    tab = pd.DataFrame(rows)
    tables["11_overview"] = tab
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(tab) + 1.4))
        y = np.arange(len(tab))
        ax.hlines(y, tab.test_sharpe_min, tab.test_sharpe_max, color="#b7d3f6", lw=4, zorder=2, label="range over seeds")
        ax.scatter(tab.test_sharpe_mean, y, s=60, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.5, zorder=3, label="mean")
        ax.axvline(0, color=INK2, lw=0.9)
        ax.annotate("cash", (0, 0.0), xycoords=("data", "axes fraction"), xytext=(3, 4), textcoords="offset points",
                    fontsize=8, color=INK2)
        if len(bh):
            v = bh.test_sharpe.iloc[0]
            ax.axvline(v, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.annotate("buy & hold", (v, 0.0), xycoords=("data", "axes fraction"), xytext=(3, 4), textcoords="offset points",
                        fontsize=8, color=INK2)
        prev = None
        for i, g in enumerate(tab.group):
            if g != prev and i:
                ax.axhline(i - 0.5, color=GRID, lw=1)
            prev = g
        ax.set_yticks(y, tab.configuration)
        ax.invert_yaxis()
        ax.set_xlabel("test Sharpe ratio (net of costs)")
        ax.set_title("11  Every reported configuration at a glance")
        ax.legend(loc="upper right")
        ax.grid(axis="y", visible=False)
        _save(fig, out / "11_overview_all_configurations.png")
    return ["11"]


# ----------------------------------------------------------------------------- build
def compile_all(L, out_name=OUT_NAME, verbose=True):
    out = Path(L.root) / out_name
    if out.exists():
        shutil.rmtree(out)
    (out / "tables").mkdir(parents=True)
    all_runs = R.load_results(L.runs)
    df = current_runs(all_runs)
    tables, welch = {}, []
    done = []
    done += fig_data(L, out, tables)
    done += fig_rnn(L, out, tables)
    done += fig_agents(df, L.runs, out, tables)
    done += fig_ablation(df, out, tables, welch)
    done += fig_rewards(df, out, tables)
    fin, final_info = fig_final(df, L.runs, out, tables)
    done += fin
    done += fig_activity(df, out, tables)
    done += fig_overview(df, out, tables)
    for name, t in tables.items():
        t.to_csv(out / "tables" / f"{name}.csv", index=False)
    (out / "README.md").write_text(_readme(df, all_runs, tables, welch, final_info), encoding="utf-8")
    if verbose:
        print(f"{len(list(out.glob('*.png')))} figures, {len(tables)} tables and README.md -> {out}")
        for p in sorted(out.glob("*.png")):
            print("  ", p.name)
    return out


# ----------------------------------------------------------------------------- README
def _t(tables, key, digits=2):
    t = tables.get(key)
    return R.df_markdown(t, digits) if t is not None and len(t) else "_not available_"


def _readme(df, all_runs, tables, welch, final_info):
    n = len(df)
    commits = ", ".join(sorted(df.commit.dropna().astype(str).unique()))
    ag = tables.get("04_agents")
    act = tables.get("10_trading_activity")
    static_line = (f"{int(act.seeds_0_or_1_trade.sum())} of {int(act.seeds.sum())} seeds made 0-1 trades and "
                   f"{int(act.seeds_le_5_trades.sum())} made 5 or fewer") if act is not None else ""
    min_p = min((w["p_value"] for w in welch), default=float("nan"))
    fin = tables.get("08_final_model")
    fname = final_info[1] if final_info else ""
    f1 = f3 = ""
    if ag is not None and len(ag):
        best = ag.sort_values("test_sharpe_mean", ascending=False).iloc[0]
        dead = [r.agent for r in ag.itertuples() if getattr(r, "seeds_with_eligible_policy", 1) == 0]
        f1 = (f"The best one ({best.agent}) averages {best['test_return_%_mean']:+.1f} % with seeds ranging from "
              f"{best['test_return_%_min']:+.0f} % to {best['test_return_%_max']:+.0f} %."
              + (f" {' and '.join(dead)} never learned to trade actively under costs." if dead else ""))
    if fin is not None and len(fin):
        f3 = f"its test returns range from {fin['test_return_%'].min():+.0f} % to {fin['test_return_%'].max():+.0f} %"
    sec = []
    sec.append(f"""# Results walkthrough

**Thesis:** A Hybrid Reinforcement Learning and Large Language Model Framework for Risk-Aware
Cryptocurrency Trading - Taimoor Ahmed, MS Data Science (SEECS-NUST).

Figures are numbered in reading order (`01` ... `11`); each section below shows one figure, the table
behind it (also in `tables/` as CSV) and how to read it. Everything here is generated from the
experiment runs by `rltrade/compile_results.py`, so the numbers match the runs exactly.

All trading results are on the **test period** (24 Jan - 6 Jun 2026, never used for any choice),
**net of trading costs** (0.10 % fee + 0.02 % slippage per side), averaged over **5 random seeds**.

## Key findings

1. **No algorithm reliably beats holding cash.** {f1} (figures 04-05).
2. **The LLM sentiment and risk signals show no measurable benefit.** None of the A/B/C/D differences
   is statistically significant (smallest p = {min_p:.2f}; figure 06).
3. **The choice of reward matters less than the seed.** The final model was chosen on validation, but
   {f3} (figures 07-09).
4. **Many trained policies barely trade.** {static_line} (figure 10). A single short held through the
   falling test period earns about +32 %, so such seeds can look skilful without trading.
5. All of this is measured with a leak-free pipeline: signals lagged by one day, causal factor model,
   realistic costs, and selection on validation only.

| # | Figure | Question |
|---|---|---|
| 01 | `01_data_btc_price_and_splits.png` | What market period is used, and how is it split? |
| 02 | `02_data_daily_llm_signals.png` | What do the LLM signals look like? |
| 03 | `03_factor_model_architectures.png` | Which factor model compresses the price features? |
| 04 | `04_agents_test_performance.png` | Which RL algorithm performs best? |
| 05 | `05_agents_equity_curves.png` | How does their equity evolve over the test period? |
| 06 | `06_ablation_llm_signals.png` | **Do the LLM signals improve trading?** (main research question) |
| 07 | `07_reward_functions.png` | Which reward function works best? |
| 08 | `08_final_model_validation_vs_test.png` | Does the selected model's validation result hold on test? |
| 09 | `09_final_model_equity_per_seed.png` | How does each seed of the final model trade? |
| 10 | `10_trading_activity.png` | How actively do the trained policies trade? |
| 11 | `11_overview_all_configurations.png` | Everything on one chart |
""")
    sec.append(f"""## 01 - Market data and splits

![01](01_data_btc_price_and_splits.png)

Bitcoin 1-minute bars from Binance's public archive (2024-01-01 to 2026-06-06, 1,278,720 bars), split
chronologically 70/15/15. Models are trained on the first period, all choices (hyperparameters,
checkpoints, algorithm, reward) are made on validation, and the test period is used once at the end.
The test period is a falling market, which matters when reading every later figure.

{_t(tables, "01_data_splits", 1)}
""")
    sec.append(f"""## 02 - LLM signals

![02](02_data_daily_llm_signals.png)

About 278k Bitcoin tweets were each scored by an LLM (DeepSeek-V3) for **sentiment** (1 bearish - 5
bullish) and **risk** (1 calm - 5 high perceived risk), then averaged per day. The agent only ever sees
the **previous day's** value (no look-ahead).

{_t(tables, "02_data_signals", 3)}
""")
    sec.append(f"""## 03 - Factor model

![03](03_factor_model_architectures.png)

101 technical "alpha" features are compressed into 8 factors by a causal recurrent network. Several
architectures were tuned with Optuna; the one with the lowest validation error (dark bar) feeds every
trading experiment. The architectures are close to each other, so the choice is not critical.

{_t(tables, "03_factor_model", 4)}
""")
    sec.append(f"""## 04 - RL algorithms

![04](04_agents_test_performance.png)

Five algorithms, each tuned with Optuna (15 trials) and then trained with 5 seeds, using all signals
(state D). Bars are the mean over seeds, dots are the individual seeds, and the dashed line is buy &
hold. A checkpoint is only eligible if it trades at least 5 times on validation. PPO and A2C never
produced such a policy, so their 0 % means "found nothing", not "chose cash".

{_t(tables, "04_agents")}
""")
    sec.append("""## 05 - Equity curves of the algorithms

![05](05_agents_equity_curves.png)

Mean cumulative return over the test period per algorithm, against buy & hold and cash. Seed-level
numbers are in table 04.
""")
    sec.append(f"""## 06 - Do the LLM signals help? (main research question)

![06](06_ablation_llm_signals.png)

The same tuned Double DQN is trained with four versions of its input: **A** prices only, **B** +
sentiment, **C** + risk, **D** + both. Everything else is identical. If the signals helped, B, C and D
would beat A consistently. They do not: the seed-to-seed spread is far larger than the differences, and
no comparison is significant at n = 5. (State D is the Double DQN run from figure 04, with the same
configuration and seeds.)

{_t(tables, "06_ablation")}

Welch t-tests on test Sharpe:

{_t(tables, "06_ablation_tests", 3)}
""")
    sec.append(f"""## 07 - Reward functions

![07](07_reward_functions.png)

Three rewards for the tuned Double DQN (state D): change in net assets, differential Sharpe, and a
drawdown penalty. The drawdown-penalised agent stays out of the market most of the time, so it has
smaller drawdowns but still loses on average.

{_t(tables, "07_rewards")}
""")
    sec.append(f"""## 08 - Final model: validation vs test

![08](08_final_model_validation_vs_test.png)

The final model ({fname}) is the configuration with the best **validation** Sharpe across all
experiments. Blue is each seed's validation Sharpe, orange its test Sharpe. Validation is good and
consistent; test is not. This gap is the central caveat of the results.

{_t(tables, "08_final_model")}
""")
    sec.append("""## 09 - Final model equity, per seed

![09](09_final_model_equity_per_seed.png)

Each seed of the final model on the test period. The best seed's result comes from a few large
positions, so it should not be read as the model's expected performance.
""")
    sec.append(f"""## 10 - Trading activity

![10](10_trading_activity.png)

The number of trades each trained policy makes over the test period. Dots in the shaded zone (5 or
fewer trades) are near-static: they take one position and hold it. In a falling market, one short
position earns about +32 % without any skill, which explains some of the best-looking seeds.

{_t(tables, "10_trading_activity", 1)}
""")
    sec.append(f"""## 11 - Overview

![11](11_overview_all_configurations.png)

Every reported configuration's test Sharpe: dot = mean over seeds, bar = range from worst to best seed.
Almost every range crosses zero (cash) and they overlap each other heavily.

{_t(tables, "11_overview")}

## Notes

- Runs shown: {n} (experiments `{EXP_AGENTS}`, `{EXP_ABLATION}`, `{EXP_REWARDS}`; code commits {commits}).
  Code: GitHub `taimoor-ahmed-1/FinAI-CryptoRaiders`, branches `exp/01-setup` ... `exp/08-results`.
- Not shown here (kept on Drive under `runs/` and `report/`): two superseded first attempts (PPO with
  default hyperparameters; a 1-trade eligibility rule), the runs that deliberately used same-day
  (look-ahead) signals to measure the leak, and the original leaky factor model.
- Limitations: one test window, in a falling market; 5 seeds per configuration; daily signals feeding
  5-minute decisions; LLM scores not yet checked against human labels.
""")
    return "\n".join(sec)
