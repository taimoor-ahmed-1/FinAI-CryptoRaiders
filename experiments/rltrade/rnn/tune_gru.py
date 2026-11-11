"""
GRU re-tune on the initiation split, with two architecture options in the search (experiment 15).

Same machinery as tune.py (Optuna TPE, median pruner, SQLite study that resumes, reduced step budget per
trial, scored by full validation-split MSE at the best checkpoint), with this search space:

    inputs        fixed_plus (115) | fixed_plus_long (+18 slow 4-hour / 1-day inputs, rltrade.long_context)
    skip          False | True     linear shortcut from the current bar's inputs to the outputs (rnn.zoo)
    seq_len       256 | 512 | 1024 | 2048   (warm-up = seq_len / 4)
    mid_dim       64 | 128 | 192 | 256,  num_layers 1-3,  dropout 0-0.5
    lr 1e-4 - 3e-3 (log), weight_decay 1e-5 - 1e-2 (log), optimizer adamw | adam | rmsprop,
    batch_size 32 | 64 | 128, label_noise 0 - 0.05

Then `final_runs` retrains the best trial with the full budget over several seeds, and `ensemble` /
`blend` score averaged predictions on validation. Only train and validation are used to choose anything.
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch as th

from .train import RnnConfig, label_metrics, train_rnn

INPUT_SETS = ("fixed_plus", "fixed_plus_long")


def suggest(trial, max_steps, max_minutes, seed=0):
    seq_len = trial.suggest_categorical("seq_len", [256, 512, 1024, 2048])
    hp = dict(mid_dim=trial.suggest_categorical("mid_dim", [64, 128, 192, 256]),
              num_layers=trial.suggest_int("num_layers", 1, 3),
              dropout=trial.suggest_float("dropout", 0.0, 0.5),
              skip=trial.suggest_categorical("skip", [False, True]))
    cfg = RnnConfig(arch="gru", hp=hp, seq_len=seq_len, wup_dim=seq_len // 4,
                    lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                    weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                    optimizer=trial.suggest_categorical("optimizer", ["adamw", "adam", "rmsprop"]),
                    batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                    label_noise=trial.suggest_float("label_noise", 0.0, 0.05),
                    max_steps=max_steps, max_minutes=max_minutes, valid_gap=50, patience=10, seed=seed)
    return trial.suggest_categorical("inputs", list(INPUT_SETS)), cfg


def from_params(params, **overrides):
    import optuna
    inputs, cfg = suggest(optuna.trial.FixedTrial(params), max_steps=0, max_minutes=0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return inputs, cfg


def tune(prepared, out_dir, n_trials=40, max_steps=3000, max_minutes=10, seed=0):
    """prepared: {input set name: prepared dir}. Runs or resumes the study until `n_trials` are finished."""
    import optuna
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="gru_e15", storage=f"sqlite:///{out / 'optuna_gru.db'}", load_if_exists=True,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=500))
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]

    def objective(trial):
        inputs, cfg = suggest(trial, max_steps, max_minutes, seed=seed)
        oom = False
        try:
            m = train_rnn(cfg, prepared[inputs], out / "trials" / f"{trial.number:03d}", trial=trial,
                          verbose=False, save_factors=False)
        except th.cuda.OutOfMemoryError:
            oom = True
        if oom:
            gc.collect()
            if th.cuda.is_available():
                th.cuda.empty_cache()
            trial.set_user_attr("oom", True)
            print(f"  [#{trial.number}] out of GPU memory (seq_len {cfg.seq_len}, batch {cfg.batch_size}) - pruned")
            raise optuna.TrialPruned("CUDA out of memory")
        for k in ("ic_mean", "dir_acc_mean"):
            trial.set_user_attr(f"val_{k}", m["val"][k])
        trial.set_user_attr("params", m["params"])
        trial.set_user_attr("minutes", m["minutes"])
        print(f"  [#{trial.number}] {inputs:15s} skip={cfg.hp['skip']!s:5s} seq {cfg.seq_len:4d} mid {cfg.hp['mid_dim']:3d} "
              f"x{cfg.hp['num_layers']} | val mse {m['val']['mse']:.5f} ic {m['val']['ic_mean']:+.3f} "
              f"({m['minutes']} min, {m['stop_reason']})")
        return m["val"]["mse"]

    remaining = n_trials - len(done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    study.trials_dataframe().to_csv(out / "trials_gru.csv", index=False)
    return study


def final_runs(params, prepared, out_dir, seeds=(0, 1, 2), max_minutes=60):
    """Full-budget retraining of one trial's configuration over several seeds. Returns run folders."""
    inputs, _ = from_params(params)
    dirs = []
    for s in seeds:
        d = Path(out_dir) / "final" / f"gru_s{s}"
        if not (d / "metrics.json").exists():
            _, cfg = from_params(params, seed=s, max_minutes=max_minutes)
            train_rnn(cfg, prepared[inputs], d)
        dirs.append(d)
    return inputs, dirs


def load_factors(run_dirs, split):
    return [np.load(Path(d) / f"factors_{split}.npy") for d in run_dirs]


def ensemble(run_dirs, labels_val):
    """Validation metrics of the mean prediction of several runs."""
    return label_metrics(np.mean(load_factors(run_dirs, "val"), axis=0), labels_val)


def blend(dirs_a, dirs_b, labels_val, weights=np.round(np.arange(0, 1.01, 0.1), 2)):
    """Validation MSE of w * mean(a) + (1 - w) * mean(b) on a grid of w. Returns (best w, rows)."""
    a = np.mean(load_factors(dirs_a, "val"), axis=0)
    b = np.mean(load_factors(dirs_b, "val"), axis=0)
    rows = []
    for w in weights:
        m = label_metrics(w * a + (1 - w) * b, labels_val)
        rows.append({"w_first": float(w), "mse": m["mse"], "ic": m["ic_mean"], "dir_acc": m["dir_acc_mean"]})
    best = min(rows, key=lambda r: r["mse"])["w_first"]
    return best, rows


def write_mixture(dirs_a, dirs_b, w, out_dir, splits):
    """Factors w * mean(a) + (1 - w) * mean(b) for every split, in the layout the trading agent reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in splits:
        a = np.mean(load_factors(dirs_a, s), axis=0)
        f = w * a + (1 - w) * np.mean(load_factors(dirs_b, s), axis=0) if dirs_b else a
        np.save(out / f"factors_{s}.npy", f.astype(np.float32))
    (out / "mixture.json").write_text(json.dumps({"first": [str(d) for d in dirs_a], "second": [str(d) for d in dirs_b],
                                                  "w_first": w}, indent=1))
    return out
