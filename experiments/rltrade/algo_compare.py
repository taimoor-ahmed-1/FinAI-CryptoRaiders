"""
Experiment 22: compare PPO, discrete SAC and RecurrentPPO with experiment 16's Double DQN on validation.

Every algorithm is tuned and trained as E16 but with a decision every 60 bars (E23a) (state A, E15 factors, 15 Optuna trials
x 500k steps scored on validation, 5 seeds x 2M steps, a checkpoint must trade >= 5 times, and the
stay-flat fallback unless it beats cash and buy and hold). Each algorithm has its own output root
under initiation/e22_algorithms/<agent>; E16's runs stay where they are (initiation/agents).

`compare` reads whatever has finished, so each notebook can print the table so far.
"""
from pathlib import Path

import numpy as np
import pandas as pd

E22_AGENTS = ("ppo", "sac", "rppo")
NAMES = {"ddqn": "Double DQN (experiment 1, 5-min decisions)", "ppo": "PPO (60-min)", "sac": "Discrete SAC (60-min)",
         "rppo": "Recurrent PPO, LSTM (60-min)"}


def sources(init_dir):
    """(agent, label, output root, experiment name) for E16's DDQN and every E22 algorithm."""
    init_dir = Path(init_dir)
    return [("ddqn", NAMES["ddqn"], init_dir / "agents", "16_agents")] + \
           [(a, NAMES[a], init_dir / "e22_algorithms" / a, f"22_{a}") for a in E22_AGENTS]


def _ms(x, digits=2, sign=True):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if x.empty:
        return ""
    f = f"{'+' if sign else ''}.{digits}f"
    return f"{x.mean():{f}} ± {x.std(ddof=1):.{digits}f}" if len(x) > 1 else f"{x.mean():{f}}"


def load_runs(init_dir):
    """Every finished agent run (one row per seed) of E16's DDQN and the E22 algorithms."""
    frames = []
    for agent, label, out, exp in sources(init_dir):
        f = out / "results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[(df.experiment == exp) & (df.agent == agent)].copy()
        df["algorithm"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare(init_dir):
    """One row per algorithm (mean ± SD over seeds) plus E16's baselines, all on validation."""
    runs = load_runs(init_dir)
    rows = []
    for label, g in (runs.groupby("algorithm", sort=False) if len(runs) else []):
        flat = g.fallback_used.astype(str).eq("True")
        rows.append({"policy": label, "seeds": len(g),
                     "val Sharpe": _ms(g.val_sharpe), "val return %": _ms(g.val_cum_return_pct, 1),
                     "val max DD %": _ms(g.val_max_drawdown_pct, 1, False), "trades": _ms(g.val_num_trades, 0, False),
                     "stayed flat": f"{int(flat.sum())} of {len(g)}",
                     "checkpoint val Sharpe (before fallback)": _ms(g.ckpt_val_sharpe)})
    # baselines at the E22 decision interval when an E22 notebook has run, else E16's (5-minute) ones
    base = [(out / "results.csv", exp) for a, _, out, exp in sources(init_dir)[1:] if (out / "results.csv").exists()]
    base = base[0] if base else (Path(init_dir) / "agents" / "results.csv", "16_agents")
    if base[0].exists():
        b = pd.read_csv(base[0])
        for name in ("buy_and_hold", "always_short", "cash"):
            r = b[(b.experiment == base[1]) & (b.agent == name)]
            if len(r):
                r = r.iloc[0]
                rows.append({"policy": name.replace("_", " "), "seeds": 1, "val Sharpe": f"{r.val_sharpe:+.2f}",
                             "val return %": f"{r.val_cum_return_pct:+.1f}",
                             "val max DD %": f"{r.val_max_drawdown_pct:.1f}", "trades": f"{int(r.val_num_trades)}",
                             "stayed flat": "", "checkpoint val Sharpe (before fallback)": ""})
    return pd.DataFrame(rows), runs


def plot_equity(init_dir, path=None):
    """Validation equity of every finished run, one panel per algorithm, with buy and hold."""
    import matplotlib.pyplot as plt
    from .results import as_datetime
    srcs = sources(init_dir)
    fig, axes = plt.subplots(1, len(srcs), figsize=(5 * len(srcs), 3.8), sharey=True)
    bh = None
    bh_f = Path(init_dir) / "agents" / "models" / "16_agents" / "buy_and_hold_A_net_asset_change_s0" / "val_backtest.npz"
    if bh_f.exists():
        z = np.load(bh_f)
        bh = as_datetime(z["time_ns"]), z["equity"][1:]
    runs = load_runs(init_dir)
    for ax, (agent, label, out, exp) in zip(axes, srcs):
        g = runs[runs.algorithm == label] if len(runs) else runs
        for _, r in g.iterrows():
            f = out / "models" / r.run_id / "val_backtest.npz"
            if f.exists():
                z = np.load(f)
                ax.plot(as_datetime(z["time_ns"]), z["equity"][1:], lw=1, alpha=0.8,
                        ls=":" if str(r.fallback_used) == "True" else "-", label=f"seed {r.seed}")
        if bh is not None:
            ax.plot(*bh, color="black", ls="--", lw=1.2, label="buy and hold")
        ax.set_title(f"{label}\n(dotted = stayed flat)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("equity, validation (start = 1)")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
