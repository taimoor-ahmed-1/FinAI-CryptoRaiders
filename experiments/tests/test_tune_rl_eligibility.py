"""Agent tuning must never pick a policy that does not trade.

In a falling validation market "stay in cash" scores Sharpe 0 and beats every
policy that trades and loses; such trials must score INVALID_SCORE.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import tune_rl          # noqa: E402
from rltrade.runner import INVALID_SCORE   # noqa: E402


def test_never_trading_trial_cannot_win():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    calls = []

    def fake_run(spec, prepared_dir, factor_dir, out_root, device="auto", trial=None, save=True, verbose=True):
        calls.append(spec.min_val_trades)
        flat = len(calls) % 2 == 1                       # odd trials: a do-nothing policy (Sharpe 0)
        return {"val_sharpe": 0.0 if flat else -0.8, "val_cum_return_pct": 0.0 if flat else -3.0,
                "val_max_drawdown_pct": 0.0, "val_num_trades": 0 if flat else 9,
                "selection_eligible": not flat, "minutes": 0.0}

    real = tune_rl.run
    tune_rl.run = fake_run
    try:
        study = tune_rl.tune_agent("ppo", "unused", "unused", tempfile.mkdtemp(), n_trials=4, total_steps=1, eval_every=1)
    finally:
        tune_rl.run = real
    values = sorted(t.value for t in study.trials)
    assert values == [INVALID_SCORE, INVALID_SCORE, -0.8, -0.8]
    assert study.best_value == -0.8 and study.best_trial.user_attrs["val_num_trades"] == 9
    assert all(m == 1 for m in calls) and study.study_name.endswith("_mt1")


if __name__ == "__main__":
    test_never_trading_trial_cannot_win()
    print("never-trading trials cannot win the search ... OK")
