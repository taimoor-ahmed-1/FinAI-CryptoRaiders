"""
RNN factor-model search (brief section 3.3).

Per architecture: an Optuna study (TPE sampler, median pruner) over hidden size,
depth, sequence length, dropout, learning rate, weight decay, optimiser, batch
size and label smoothing. Every trial trains with a reduced step budget and is
scored on the VALIDATION split (full-split MSE at its best checkpoint). Studies
live in a SQLite file, so an interrupted Colab session resumes where it stopped
and every trial stays logged.

Then `final_runs` retrains each architecture's best configuration with the full
budget over several seeds, and `select` picks the architecture with the lowest
mean validation MSE; its best-validation seed provides the factors for the RL
experiments. Test metrics are reported alongside but never used.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from .train import RnnConfig, train_rnn

SEARCH_ARCHS = ("lstm", "gru", "enhanced", "transformer", "tcn")    # enhanced = the LSTM+GRU hybrid


def suggest_config(trial, arch, max_steps, max_minutes, seed=0):
    """Map an Optuna trial to an RnnConfig. Search space per architecture + shared training knobs."""
    hp = {}
    if arch == "enhanced":
        heads = trial.suggest_categorical("num_heads", [2, 4, 6, 8])
        hp.update(num_heads=heads, mid_dim=heads * trial.suggest_categorical("head_dim", [16, 24, 32]),
                  num_layers=trial.suggest_int("num_layers", 1, 4),
                  num_blocks=trial.suggest_int("num_blocks", 1, 3))
    elif arch in ("lstm", "gru"):
        hp.update(mid_dim=trial.suggest_categorical("mid_dim", [64, 128, 192, 256]),
                  num_layers=trial.suggest_int("num_layers", 1, 4))
    elif arch == "transformer":
        heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        hp.update(num_heads=heads, mid_dim=heads * trial.suggest_categorical("head_dim", [16, 32]),
                  num_layers=trial.suggest_int("num_layers", 1, 4))
    elif arch == "tcn":
        hp.update(mid_dim=trial.suggest_categorical("mid_dim", [64, 128, 192]),
                  num_layers=trial.suggest_int("num_layers", 4, 9),
                  kernel=trial.suggest_categorical("kernel", [2, 3, 5]))
    else:
        raise ValueError(f"no search space for {arch!r}")
    hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    seq_len = trial.suggest_categorical("seq_len", [256, 512, 1024])
    return RnnConfig(
        arch=arch, hp=hp, seq_len=seq_len, wup_dim=seq_len // 4,
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        optimizer=trial.suggest_categorical("optimizer", ["adamw", "adam", "rmsprop"]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        label_noise=trial.suggest_float("label_noise", 0.0, 0.05),
        max_steps=max_steps, max_minutes=max_minutes, valid_gap=50, patience=10, seed=seed)


def config_from_params(arch, params, **overrides):
    """Rebuild the RnnConfig of a finished trial from its stored params."""
    import optuna
    trial = optuna.trial.FixedTrial(params)
    cfg = suggest_config(trial, arch, max_steps=0, max_minutes=0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def tune_arch(arch, prepared_dir, out_dir, n_trials=20, max_steps=3000, max_minutes=15, seed=0):
    """Run (or resume) the study for one architecture until it has `n_trials` finished trials."""
    import optuna
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"rnn_{arch}", storage=f"sqlite:///{out / 'optuna_rnn.db'}", load_if_exists=True,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=500))
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                   optuna.trial.TrialState.PRUNED)]

    def objective(trial):
        cfg = suggest_config(trial, arch, max_steps, max_minutes, seed=seed)
        m = train_rnn(cfg, prepared_dir, out / "trials" / arch / f"{trial.number:03d}",
                      trial=trial, verbose=False, save_factors=False)
        trial.set_user_attr("val_ic", m["val"]["ic_mean"])
        trial.set_user_attr("val_dir_acc", m["val"]["dir_acc_mean"])
        trial.set_user_attr("params", m["params"])
        trial.set_user_attr("minutes", m["minutes"])
        print(f"  [{arch} #{trial.number}] val mse {m['val']['mse']:.5f} ic {m['val']['ic_mean']:+.3f} "
              f"({m['minutes']} min, {m['stop_reason']})")
        return m["val"]["mse"]

    remaining = n_trials - len(done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    study.trials_dataframe().to_csv(out / f"trials_{arch}.csv", index=False)
    return study


def final_runs(arch, params, prepared_dir, out_dir, seeds=(0, 1, 2), **overrides):
    """Full-budget training of one configuration over several seeds."""
    rows = []
    for s in seeds:
        run_dir = Path(out_dir) / "final" / f"{arch}_s{s}"
        if (run_dir / "metrics.json").exists():
            m = json.loads((run_dir / "metrics.json").read_text())
        else:
            cfg = config_from_params(arch, params, seed=s, **overrides) if params else RnnConfig(arch=arch, seed=s)
            m = train_rnn(cfg, prepared_dir, run_dir)
        rows.append({"arch": arch, "seed": s, "dir": str(run_dir), "params": m["params"],
                     "val_mse": m["val"]["mse"], "val_ic": m["val"]["ic_mean"], "val_dir_acc": m["val"]["dir_acc_mean"],
                     "test_mse": m["test"]["mse"], "test_ic": m["test"]["ic_mean"],
                     "test_dir_acc": m["test"]["dir_acc_mean"], "minutes": m["minutes"]})
    return rows


def select(rows, out_dir):
    """Lowest mean validation MSE across seeds wins; its best-validation seed supplies the factors."""
    by = {}
    for r in rows:
        by.setdefault(r["arch"], []).append(r)
    board = sorted(({"arch": a, "val_mse_mean": float(np.mean([r["val_mse"] for r in rs])),
                     "val_mse_std": float(np.std([r["val_mse"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0,
                     "val_ic_mean": float(np.mean([r["val_ic"] for r in rs])),
                     "test_mse_mean": float(np.mean([r["test_mse"] for r in rs])),
                     "test_ic_mean": float(np.mean([r["test_ic"] for r in rs])), "seeds": len(rs)}
                    for a, rs in by.items()), key=lambda d: d["val_mse_mean"])
    win = board[0]["arch"]
    best = min(by[win], key=lambda r: r["val_mse"])
    sel = {"arch": win, "seed": best["seed"], "factor_dir": best["dir"], "val_mse": best["val_mse"],
           "rule": "lowest mean validation MSE over seeds; best-validation seed of that architecture"}
    out = Path(out_dir)
    (out / "selected.json").write_text(json.dumps(sel, indent=2))
    (out / "leaderboard.json").write_text(json.dumps(board, indent=2))
    return sel, board


def copy_selected(out_dir, dest):
    """Copy the selected run (factors, weights, config) to a stable location."""
    sel = json.loads((Path(out_dir) / "selected.json").read_text())
    shutil.copytree(sel["factor_dir"], dest, dirs_exist_ok=True)
    return dest
