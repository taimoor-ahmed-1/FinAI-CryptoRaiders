# Look-ahead leakage audit (brief §3.3 — hard gate)

**Trigger.** In earlier runs the RNN's validation loss kept falling throughout training.
The brief requires checking factor lag alignment, label construction and window
boundaries against the split before any tuned result is reported.

**Verdict.** Leakage was present, through **six independent paths** — four in data
preparation (same-day LLM signals, per-split Alpha101 normalisation, per-split LLM
min-max scaling, `bfill`) and two inside the RNN (unmasked attention, chained hidden
state) — plus one bug that made validation meaningless. All are fixed on branch `exp/02-leakage-fixes` and
covered by tests that fail if they return. Every result in this project comes from
the fixed code.

## What was checked, and what was found

| Check | Finding | Status | Evidence |
|---|---|---|---|
| **Label construction** (`seq_to_label`) | The definition is sound: `label[t]` compares the smoothed price at *t* with the smoothed price at *t + offset*. It *must* look forward — it is the target, never an input. The last `max_offset` = 900 rows of each split have no label and are excluded from the loss (they were zero-padded). | OK after fix | `labels.py` |
| **Label ↔ split alignment** | Validation/test targets were the first rows of the **training** labels (`SeqData` fallback). Not a leak, but validation loss had nothing to do with the validation inputs, so early stopping and "best model" were arbitrary. This alone can explain a validation curve that does not behave like one. | **Fixed** | `CHANGES.md` #7 |
| **Model causality — attention** | Unmasked multi-head attention: every step attended to the whole 512-bar window, future included. | **Fixed** | `tests/test_causality.py` |
| **Model causality — recurrent state** | Block *i*'s final hidden state (having read the whole window) initialised block *i+1*: output at step 0 depended on step 511. Masking attention alone reduced the measured leak ~50× but left it non-zero. | **Fixed** | `tests/test_causality.py` (fixed model: exactly 0; original: ~1.2) |
| **Factor-writing path** | The factors fed to the agent were produced by the leaky model, carrying state chunk to chunk. | **Fixed** | `tests/test_causality.py::test_factor_prediction_is_causal` |
| **Factor lag alignment** (Alpha101) | All 101 formulas use only current and past bars (`delay`, `ts_*`, rolling windows). `decay_linear` contained a `bfill` that copied **future** values into gaps. | **Fixed** | `tests/test_alpha101_equivalence.py` |
| **Normalisation** | Alpha101 quantile bounds were fitted on each split separately — the test split was scaled with its own future distribution. LLM signals were min-max scaled over the loaded split (the evaluator: over the evaluation split). | **Fixed**: train-fitted bounds; fixed 1–5 scale for LLM scores | `CHANGES.md` #2, #21 |
| **Window boundaries vs split** | Rolling factors restarted at each split boundary (warm-up zeros, not a leak). RNN validation windows may begin up to `wup_dim` bars *before* the validation split — that is past data, used only as warm-up context, never scored. Training windows never extend past the labelled training rows. | OK after fix | `rnn/train.py` (`valid_starts`, `sample_train`) |
| **LLM signal timing** | Daily tweet aggregates were joined to bars **of the same day**: a bar at 00:05 saw tweets posted until 23:59. This is look-ahead of up to 24 h in exactly the signals under study (RQ1–RQ3). | **Fixed**: one-day lag at preparation; dataset untouched | `tests/test_signal_lag.py` |
| **Execution timing** | Trades filled at the midpoint of the bar just observed; that midpoint is (high+low)/2 of the same bar. | **Fixed**: fill at the next bar | `tests/test_env.py` |
| **Split** | Chronological 70/15/15, no shuffling, no overlap. Test is never used for training, early stopping, checkpoint choice or hyperparameter search. | OK | `prepare.split_bounds`, `runner.py`, `tune_rl.py`, `rnn/tune.py` |

## How to re-verify (takes ~1 minute)

```bash
cd experiments
python tests/test_causality.py
BTC_CSV=/path/to/BTC_1min_with_sentiment_risk_train.csv.gz python tests/test_signal_lag.py
BTC_CSV=/path/to/BTC_1min_with_sentiment_risk_train.csv.gz python tests/test_alpha101_equivalence.py
python tests/test_env.py
```

Notebook `01_prepare_data` runs the first three automatically and stops if one fails.

## To complete after the runs (fill in)

1. **RNN validation curve with the fixed pipeline** (`02_rnn_train`, figure `rnn_default_loss.png`):
   does validation loss now flatten or turn up and trigger early stopping? Best step: ___, stop reason: ___.
2. **Leak demonstration** (`02_rnn_train`, `RUN_LEAK_DEMO = True`): validation MSE / IC of
   `enhanced` ___ vs `enhanced_legacy` ___. A clearly better "legacy" score is information from the future.
3. **Size of the sentiment look-ahead** (`05_rl_ablation`, `LEAK_CHECK = True`): test Sharpe of
   configuration D with lagged signals ___ ± ___ vs same-day signals ___ ± ___.
