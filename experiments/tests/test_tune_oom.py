"""An oversized RNN configuration must not stop the search.

On the T4, one hybrid-model trial (attention over 1024 steps, batch 128) ran out of
GPU memory and the exception ended the whole Optuna study. Such trials must be
pruned, with memory freed, and the search must continue.
"""
import sys
import tempfile
from pathlib import Path

import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade.rnn import tune   # noqa: E402


def test_out_of_memory_trial_is_pruned_and_search_continues():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    calls = []

    def fake_train(cfg, prepared_dir, out_dir, trial=None, verbose=True, save_factors=True):
        calls.append(cfg.seq_len)
        if len(calls) % 2 == 1:                      # every other trial "runs out of memory"
            raise th.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB.")
        m = {"mse": 0.2 - 0.01 * len(calls), "ic_mean": 0.1, "dir_acc_mean": 0.55}
        return {"val": m, "params": 1, "minutes": 0.0, "stop_reason": "step budget"}

    real = tune.train_rnn
    tune.train_rnn = fake_train
    try:
        study = tune.tune_arch("gru", "unused", tempfile.mkdtemp(), n_trials=4, max_steps=1, max_minutes=0)
    finally:
        tune.train_rnn = real
    states = [t.state for t in study.trials]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(calls) == 4 and len(states) == 4
    assert len(pruned) == 2 and all(t.user_attrs.get("oom") for t in pruned)
    assert study.best_value < 0.2                     # the healthy trials still count


if __name__ == "__main__":
    test_out_of_memory_trial_is_pruned_and_search_continues()
    print("out-of-memory trials pruned, search continued ... OK")
