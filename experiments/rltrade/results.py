"""
Aggregation, tables and plots for the report (brief sections 4.2 and 4.3).

Everything reads what the runner wrote: `<out>/results.csv` (one row per run) and
`<out>/models/<run_id>/test_backtest.npz` (test equity curve per run).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import drawdown_curve

GROUP = ["experiment", "agent", "state", "signal_source", "reward", "tag"]
HEADLINE = ["test_cum_return_pct", "test_sharpe", "test_sortino", "test_max_drawdown_pct", "test_romad",
            "test_win_rate_pct", "test_num_trades", "test_turnover", "test_exposure_pct"]
LABELS = {"test_cum_return_pct": "Return %", "test_sharpe": "Sharpe", "test_sortino": "Sortino",
          "test_max_drawdown_pct": "Max DD %", "test_romad": "RoMaD", "test_win_rate_pct": "Win %",
          "test_num_trades": "Trades", "test_turnover": "Turnover", "test_exposure_pct": "In market %",
          "val_sharpe": "Val Sharpe"}


def load_results(out_root):
    df = pd.read_csv(Path(out_root) / "results.csv")
    df["tag"] = df["tag"].fillna("")
    return df


def summarise(df, metrics=HEADLINE, group=GROUP):
    """mean, std and seed count per configuration."""
    g = df.groupby(group, dropna=False)
    out = g[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["seeds"] = g.size()
    return out.reset_index()


def markdown_table(summary, metrics=HEADLINE, group=("agent", "state", "reward", "tag"), digits=2):
    """Human-readable 'mean ± std' table (no extra dependency)."""
    group = [c for c in group if c in summary.columns and summary[c].astype(str).str.len().gt(0).any()]
    head = group + [LABELS.get(m, m) for m in metrics] + ["Seeds"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in group]
        for m in metrics:
            mu, sd = r[f"{m}_mean"], r[f"{m}_std"]
            if pd.isna(mu):
                cells.append("–")                  # e.g. win rate of a policy that never traded
                continue
            cells.append(f"{mu:.{digits}f}" + ("" if pd.isna(sd) else f" ± {sd:.{digits}f}"))
        cells.append(str(int(r["seeds"])))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def df_markdown(df, digits=4):
    """Any DataFrame as a Markdown table (no `tabulate` dependency)."""
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _curves(out_root, run_ids, key="equity"):
    arrs, t = [], None
    for rid in run_ids:
        f = Path(out_root) / "models" / rid / "test_backtest.npz"
        if f.exists():
            z = np.load(f)
            arrs.append(z[key] if key != "drawdown" else drawdown_curve(z["equity"]))
            t = z["time_ns"]
    if not arrs:
        return None, None
    n = min(len(a) for a in arrs)
    return t[:n - 1] if t is not None else None, np.stack([a[:n] for a in arrs])


def plot_curves(df, out_root, groups, kind="equity", title=None, ax=None, path=None):
    """Mean test equity (or drawdown) per configuration, shaded band = min..max over seeds.

    groups: {legend label: DataFrame subset (one row per seed)}
    """
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, sub in groups.items():
        t, a = _curves(out_root, sub["run_id"].tolist(), "drawdown" if kind == "drawdown" else "equity")
        if a is None:
            continue
        x = pd.to_datetime(np.concatenate([[t[0]], t]), utc=True) if t is not None else np.arange(a.shape[1])
        x = x[:a.shape[1]]
        y = (a - 1.0) * 100 if kind == "equity" else a * 100
        ax.plot(x, y.mean(0), lw=1.3, label=f"{label} (n={len(a)})")
        if len(a) > 1:
            ax.fill_between(x, y.min(0), y.max(0), alpha=0.15)
    ax.set_ylabel("cumulative return %" if kind == "equity" else "drawdown %")
    ax.set_title(title or ("Test-period equity, net of costs" if kind == "equity" else "Test-period drawdown"))
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    if own:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
    return ax


def plot_ablation(summary, path=None, title="Ablation over the state vector (test, mean ± std over seeds)"):
    import matplotlib.pyplot as plt
    s = summary.sort_values("state")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, lab in ((axes[0], "test_sharpe", "Sharpe"), (axes[1], "test_max_drawdown_pct", "Max drawdown %")):
        ax.bar(s["state"], s[f"{m}_mean"], yerr=s[f"{m}_std"].fillna(0), capsize=5,
               color=["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"][:len(s)])
        ax.set_title(lab)
        ax.set_xlabel("configuration  (A price-only, B +sentiment, C +risk, D +both)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_rnn_history(run_dirs, path=None):
    """Training / validation loss per RNN run, annotated with the leakage checks."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    notes = []
    for label, d in run_dirs.items():
        h = json.loads((Path(d) / "history.json").read_text())
        if not h:
            continue
        st = [r["step"] for r in h]
        ax.plot(st, [r["train_loss"] for r in h], lw=0.8, alpha=0.5, label=f"{label} train")
        ax.plot(st, [r["val_loss"] for r in h], lw=1.5, label=f"{label} val")
        vl = np.array([r["val_loss"] for r in h])
        best = int(np.argmin(vl))
        monotone = bool(np.all(np.diff(vl) <= 0))
        notes.append(f"{label}: best val at step {st[best]}, "
                     f"{'monotone decrease (!)' if monotone else 'not monotone'}")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE")
    ax.set_title("RNN loss curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.text(0.01, 0.02, "Leakage checks: causal models (tests/test_causality.py); labels per split with "
            "train-fitted thresholds;\nfactor prediction path causal; LLM signals lagged one day.\n"
            + "\n".join(notes), transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_optuna(study, out_prefix):
    """Parameter importance + parallel coordinates (skipped gracefully if unavailable)."""
    import matplotlib.pyplot as plt
    try:
        from optuna.visualization import matplotlib as ov
    except ImportError:
        return []
    saved = []
    for name, fn in (("importance", ov.plot_param_importances), ("parallel", ov.plot_parallel_coordinate),
                     ("history", ov.plot_optimization_history)):
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            fig.set_size_inches(10, 5)
            fig.tight_layout()
            fig.savefig(f"{out_prefix}_{name}.png", dpi=130)
            plt.close(fig)
            saved.append(f"{out_prefix}_{name}.png")
        except Exception as e:                               # e.g. too few finished trials
            print(f"  optuna {name} plot skipped: {e}")
    return saved
