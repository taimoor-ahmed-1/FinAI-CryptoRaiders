"""
Experiment 20a: out-of-sample ("cross-fitted") factors for the TRAIN period.

The E15 GRU was fitted on all of train, so its train-period forecasts are in-sample: the trading agent
learned on forecasts that are better than any it will see later, and train cannot serve as a
walk-forward evaluation period. Here train is cut into K consecutive blocks. For block k the E15
configuration is retrained (same hyperparameters, same seeds, early stopping on validation as in E15)
on train WITHOUT block k and without an embargo of `embargo` bars on either side of it, so no training
label looks into the block (a label at row t uses prices up to t + 900 bars). Block k's factors are then
predicted by those models only. The other splits keep the E15 factors (selected_mean3).

Purged K-fold, not a pure walk-forward: the model for an early block has also seen LATER train blocks.
Validation, test and stress_bear forecasts are unchanged.
"""
import json
from pathlib import Path

import numpy as np

from .train import train_rnn
from .tune_gru import from_params

EMBARGO = 1440                     # bars kept out on each side of a block (labels look 900 bars ahead)


def blocks(n, k):
    """K consecutive [lo, hi) row blocks covering 0..n."""
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def exclusion(block, n, embargo=EMBARGO):
    lo, hi = block
    return (max(lo - embargo, 0), min(hi + embargo, n))


def run_dir(out_dir, fold, seed):
    return Path(out_dir) / "runs" / f"fold{fold}_s{seed}"


def crossfit_runs(params, prepared_dir, out_dir, n_train, k=4, seeds=(0, 1, 2), embargo=EMBARGO, max_minutes=60,
                  verbose=True):
    """Train every (fold, seed) model that is not finished yet. Returns {fold: [run dirs]}."""
    out = {}
    for f, b in enumerate(blocks(n_train, k)):
        out[f] = []
        for s in seeds:
            d = run_dir(out_dir, f, s)
            if not (d / "metrics.json").exists():
                _, cfg = from_params(params, seed=s, max_minutes=max_minutes)
                cfg.train_exclude = (exclusion(b, n_train, embargo),)
                if verbose:
                    print(f"fold {f} (train rows {b[0]:,}-{b[1]:,} held out, {cfg.train_exclude[0]} excluded), seed {s}")
                train_rnn(cfg, prepared_dir, d, verbose=verbose, save_splits=("train",))
            out[f].append(d)
    return out


def write_factors(fold_dirs, n_train, base_factor_dir, out_dir, splits, k=4, embargo=EMBARGO):
    """factors_train = each block predicted by the mean of its fold's models; other splits copied from
    `base_factor_dir`. Writes the layout the trading code reads plus crossfit.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bl = blocks(n_train, k)
    tr = None
    for f, (lo, hi) in enumerate(bl):
        pred = np.mean([np.load(Path(d) / "factors_train.npy", mmap_mode="r")[lo:hi] for d in fold_dirs[f]], axis=0)
        if tr is None:
            tr = np.empty((n_train, pred.shape[1]), dtype=np.float32)
        tr[lo:hi] = pred
    np.save(out / "factors_train.npy", tr)
    for s in splits:
        if s != "train":
            np.save(out / f"factors_{s}.npy", np.load(Path(base_factor_dir) / f"factors_{s}.npy"))
    (out / "crossfit.json").write_text(json.dumps({
        "blocks": bl, "embargo": embargo, "base_factor_dir": str(base_factor_dir),
        "runs": {str(f): [str(d) for d in v] for f, v in fold_dirs.items()},
        "rule": "train block k predicted by models trained without block k +- embargo; other splits = base"}, indent=1))
    return out
