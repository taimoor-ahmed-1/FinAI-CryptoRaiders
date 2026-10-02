"""
Per-agent hyperparameter search (brief section 3.4 b: "tune each agent's own
hyperparameters - learning rate, entropy coefficient, clip range, discount, GAE
lambda, network width, rollout length, reward scaling").

Each trial trains one agent on configuration D with a reduced budget and is
scored by the VALIDATION Sharpe of its best checkpoint. Trial seeds (1000 + trial
number) never overlap the reporting seeds (0-4), and trials are not written to
results.csv - only the final multi-seed runs are reported. Studies are stored in
SQLite, so tuning resumes after a disconnect.

A trial whose best eligible checkpoint never trades on validation scores
INVALID_SCORE (see runner.selection_score): otherwise "do nothing" (Sharpe 0) would
win the search in a falling validation market.
"""
import dataclasses
from pathlib import Path

from .env import EnvConfig
from .runner import INVALID_SCORE, RunSpec, run

NETS = {"small": (128, 128), "medium": (256, 256, 128), "notebook": (256, 256, 128, 128, 64)}


def suggest(trial, agent):
    """Returns (agent_hp, env_overrides) for one trial."""
    hp = {"lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
          "gamma": trial.suggest_categorical("gamma", [0.95, 0.99, 0.995]),
          "net_dims": NETS[trial.suggest_categorical("net", list(NETS))]}
    if agent == "ppo":
        hp.update(entropy_coef=trial.suggest_float("entropy_coef", 1e-4, 0.1, log=True),
                  clip=trial.suggest_categorical("clip", [0.1, 0.2, 0.3]),
                  gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
                  rollout_len=trial.suggest_categorical("rollout_len", [32, 64, 128]),
                  minibatch=trial.suggest_categorical("minibatch", [64, 256, 1024]),
                  epochs=trial.suggest_categorical("epochs", [3, 6, 10]))
    elif agent == "a2c":
        hp.update(entropy_coef=trial.suggest_float("entropy_coef", 1e-4, 0.1, log=True),
                  gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 1.0]),
                  rollout_len=trial.suggest_categorical("rollout_len", [5, 16, 32]))
    elif agent in ("dqn", "ddqn", "d3qn"):
        hp.update(batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
                  target_update=trial.suggest_categorical("target_update", [500, 2000, 8000]),
                  eps_fraction=trial.suggest_float("eps_fraction", 0.1, 0.5),
                  replay_ratio=trial.suggest_categorical("replay_ratio", [0.125, 0.25, 0.5]))
    else:
        raise ValueError(f"no search space for {agent!r}")
    env = {"reward_scale": trial.suggest_categorical("reward_scale", [10.0, 100.0, 1000.0])}
    return hp, env


def params_to_hp(agent, params):
    import optuna
    return suggest(optuna.trial.FixedTrial(params), agent)


def tune_agent(agent, prepared_dir, factor_dir, out_root, env_cfg=None, n_trials=20,
               total_steps=500_000, eval_every=100_000, device="auto", sampler_seed=0, min_val_trades=1):
    import optuna
    env_cfg = env_cfg or EnvConfig(state="D")
    out = Path(out_root) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        # _mt<k>: the eligibility rule is part of the objective, so it is part of the study's identity
        study_name=f"rl_{agent}_{env_cfg.state}_{env_cfg.reward}_mt{min_val_trades}",
        storage=f"sqlite:///{out / 'optuna_rl.db'}",
        load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=eval_every))
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                   optuna.trial.TrialState.PRUNED)]

    def objective(trial):
        hp, env_over = suggest(trial, agent)
        spec = RunSpec("tuning", agent, seed=1000 + trial.number,
                       env=dataclasses.replace(env_cfg, **env_over), agent_hp=hp,
                       total_steps=total_steps, eval_every=eval_every, min_val_trades=min_val_trades)
        row = run(spec, prepared_dir, factor_dir, out_root, device=device, trial=trial, save=False, verbose=False)
        for k in ("val_sharpe", "val_cum_return_pct", "val_max_drawdown_pct", "val_num_trades", "selection_eligible"):
            trial.set_user_attr(k, row[k])
        eligible = bool(row["selection_eligible"])
        print(f"  [{agent} #{trial.number}] val sharpe {row['val_sharpe']:+.3f} "
              f"ret {row['val_cum_return_pct']:+.2f}% trades {row['val_num_trades']} ({row['minutes']} min)"
              + ("" if eligible else "  <- never traded: not eligible"))
        return row["val_sharpe"] if eligible else INVALID_SCORE

    remaining = n_trials - len(done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    study.trials_dataframe().to_csv(out / f"trials_{agent}.csv", index=False)
    return study
