"""
Experiment 20b: walk-forward evaluation of the factor-rule benchmark and the signal-gated RL agent.

Windows (consecutive): T1-T4 = the 4 train blocks of the cross-fit (rnn/crossfit.py, out-of-sample
factors), V1-V2 = the two halves of validation (E15 factors). Four folds, each judged on a window that
nothing in that fold was chosen or trained on:

    fold  agent trains on   agent checkpoint chosen on   rule chosen on      judged on
    1     T1                T2                           T1-T2               T3
    2     T1-T2             T3                           T1-T3               T4
    3     T1-T3             T4                           T1-T4               V1
    4     T1-T4             V1                           T1-T4 + V1          V2

Benchmark: the threshold rule (factor_rule.py) with the best net Sharpe on everything before the judged
window (at least `min_trades` trades), thresholds = quantiles of |signal| on that history.
Agent: Double DQN, state A, whose entries are gated by that same rule (EnvConfig.entry_signal /
entry_threshold / min_hold_bars): it may only open a position in the direction of an extreme reading
of the rule's factor and must keep it for the rule's minimum hold; it decides whether to take a signal
and when to get out.

Each fold is written as its own small prepared + factor directory (market_{train,val,test,hist}.npz and
factors_*.npy, sliced from the real splits) so runner.run works unchanged: its "val" is the checkpoint
window and its "test" the judged window. T4 -> V1 joins train and validation across the one missing
day between the splits (2025-01-31).
"""
import json
from pathlib import Path

import numpy as np

from . import factor_rule as FR
from .rnn.crossfit import blocks

FOLDS = (  # (agent train windows, checkpoint window, judged window)
    (("T1",), "T2", "T3"),
    (("T1", "T2"), "T3", "T4"),
    (("T1", "T2", "T3"), "T4", "V1"),
    (("T1", "T2", "T3", "T4"), "V1", "V2"),
)


def windows(n_train, n_val, k_train=4, k_val=2):
    """{name: (split, lo, hi)} for T1..Tk over train (the cross-fit blocks) and V1..Vk over validation."""
    w = {f"T{i + 1}": ("train", lo, hi) for i, (lo, hi) in enumerate(blocks(n_train, k_train))}
    w.update({f"V{i + 1}": ("val", lo, hi) for i, (lo, hi) in enumerate(blocks(n_val, k_val))})
    return w


def _concat(arrays_by_window, names):
    return np.concatenate([arrays_by_window[n] for n in names])


def write_fold(prepared_dir, factor_dir, wins, fold, out_dir):
    """Writes fold `fold` (index into FOLDS) as splits train / val / test / hist in `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_w, sel_w, eval_w = FOLDS[fold]
    parts = {"train": train_w, "val": (sel_w,), "test": (eval_w,), "hist": train_w + (sel_w,)}
    src_m = {s: np.load(Path(prepared_dir) / f"market_{s}.npz") for s in ("train", "val")}
    src_f = {s: np.load(Path(factor_dir) / f"factors_{s}.npy", mmap_mode="r") for s in ("train", "val")}
    keys = [k for k in src_m["train"].files if k in src_m["val"].files]
    for name, ws in parts.items():
        m, f = {}, []
        for k in keys:
            pieces = []
            for w in ws:
                s, lo, hi = wins[w]
                a = src_m[s][k]
                pieces.append(a[lo:hi] if a.ndim and len(a) == len(src_f[s]) else a)
            m[k] = np.concatenate(pieces) if pieces[0].ndim else pieces[0]
        for w in ws:
            s, lo, hi = wins[w]
            f.append(np.asarray(src_f[s][lo:hi], np.float32))
        np.savez(out / f"market_{name}.npz", **m)
        np.save(out / f"factors_{name}.npy", np.concatenate(f))
    (out / "fold.json").write_text(json.dumps({"fold": fold + 1, "train": train_w, "checkpoint": sel_w,
                                               "judged": eval_w, "rule_history": parts["hist"],
                                               "windows": {w: wins[w] for w in set(sum(parts.values(), ()))}},
                                              indent=1))
    return out


def select_rule(train_metrics, min_trades=5, metric="sharpe"):
    """Index of the best rule on the history (net), among rules with at least `min_trades` trades."""
    score = np.where(train_metrics["num_trades"] >= min_trades, train_metrics[metric], -np.inf)
    if not np.isfinite(score).any():
        raise ValueError("no rule trades enough on the history")
    return int(np.argmax(score))


def gate_config(rule):
    """EnvConfig overrides that gate an agent's entries with `rule` (a factor_rule grid entry)."""
    return {"entry_signal": int(rule["j"]), "entry_threshold": float(rule["k"]), "min_hold_bars": int(rule["min_hold"])}


def stitch(backtests):
    """Joins the judged windows' backtests into one out-of-sample record (each window starts flat)."""
    out = {k: np.concatenate([b[k] for b in backtests]) for k in ("returns", "level", "traded", "cost", "time_ns")}
    out["equity"] = np.concatenate([[1.0], np.cumprod(1.0 + out["returns"])])
    return out


__all__ = ["FOLDS", "windows", "write_fold", "select_rule", "gate_config", "stitch", "FR"]
