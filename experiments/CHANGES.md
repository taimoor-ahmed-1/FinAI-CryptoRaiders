# CHANGES — what differs from the provided notebooks, and why

Every change to the pipeline's behaviour is listed here with the evidence behind it.
The provided notebooks are kept unmodified in `original/` so each claim can be checked
against the code; function and variable names refer to that code.

Severity: **L** = look-ahead leakage (future information reaches the model or the
agent), **B** = bug that changes results, **R** = realism of the backtest, **Q** =
quality or speed with no intended change to results.

> **Consequence for earlier results.** Items 1–4, 7–8 and 12–15 each affect results
> produced with the provided notebooks. Several of them inflate performance: the
> leaks, the frictionless costs, and evaluating on validation prices from the wrong
> period. Earlier PoC numbers should not be compared with results from this code, and
> should not be cited as evidence.

## Data preparation — `2_data_splitter`, `3_alpha_signals_generator`

| # | Sev | Original behaviour | Effect | Fix | Verified by |
|---|---|---|---|---|---|
| 1 | **L** | Daily sentiment/risk aggregates are joined onto every 1-minute bar **of the same UTC day** (dataset construction: daily aggregates left-joined by UTC date) | The bar at 00:05 carries the mean of tweets posted up to 23:59 that day — up to 24 h of future information, in exactly the features the ablation measures | `prepare.lag_daily_signals`: every bar of day D sees day D−1's aggregate. The dataset is **not** modified; same-day values are kept as `sameday__<col>` only to measure the leak (`signal_source="sameday"`) | `tests/test_signal_lag.py` (synthetic + real data) |
| 2 | **L** | Alpha101 quantile normalisation (1 %/99 %) fitted **separately on each split**, test included | The test inputs are scaled with the test period's own future distribution | Quantiles fitted on train only, applied to all splits (`prepare.py`) | code review; `alpha101_norm.npz` stores the train bounds |
| 3 | **L** | `decay_linear` calls `df.bfill()` before `ffill()` | Leading gaps are filled with **later** values | `bfill` removed; remaining gaps → 0 | `tests/test_alpha101_equivalence.py::test_decay_linear` |
| 4 | B | Factors computed separately per split file | Every rolling window restarts at the split boundary, so the first few hundred bars of validation and test carry warm-up zeros | Computed once on the continuous series (all operators look backwards), then split | — |
| 5 | Q | `ts_rank`, `rank`, `product`, `ts_argmax/argmin`, `decay_linear` use pandas `rolling().apply` / row-wise `iloc` | Hours on 1.28 M bars | Vectorised NumPy versions reproducing the originals exactly, incl. tie handling, NaN windows, pandas' ±inf-as-missing rule and `decay_linear`'s dtype quirk (a boolean input stays boolean — alpha092) | `tests/test_alpha101_equivalence.py`: each helper, and **all 101 factors** end-to-end on real bars, identical to the originals |
| 6 | Q | Raw factors kept in float64 | — | Kept (a float32 cast overflowed a few factors to ±inf and broke their quantiles; caught in testing) | assertion in `apply_quantiles` |

## RNN factor model — `4_rnn_trainer`

| # | Sev | Original behaviour | Effect | Fix | Verified by |
|---|---|---|---|---|---|
| 7 | **B** | `SeqData` is given no validation/test labels, so it falls back to `valid_label_seq = train_label_seq[:valid_len]` (same for test) | Validation and test **targets are the first rows of the training labels** while the inputs come from the validation/test periods. Validation loss, early stopping, the chosen checkpoint and test loss are all meaningless | Labels are built for every split from its own prices; class thresholds fitted on train and reused (`labels.py`, `prepare.py`) | `labels.py` docstring; per-split `labels_*.npy` |
| 8 | **L** | `EnhancedRnnRegNet`: `self.attention(x)` with **no mask** | Every output step attends to the whole window, future bars included | Causal (lower-triangular) mask | `tests/test_causality.py` |
| 9 | **L** | `EnhancedRnnRegNet`: `x, hidden = block(x, hidden)` hands block *i*'s **final** state (which has read the whole window) to block *i+1* as its initial state | Block *i+1*'s output at step 0 depends on the last input. This path bypasses attention, so masking alone does not fix it (the test shows the leak shrinking ~50× but not vanishing) | Each block starts from a fresh state (`CausalEnhancedRnnRegNet`: same layers, same weights) | `tests/test_causality.py`: all experiment architectures change past outputs by exactly 0; the original by ~1.2 |
| 10 | B | Training labels are 900 rows shorter than the inputs and are zero-padded | The model is trained towards 0 on the last 900 training rows | Unlabelled rows are excluded from the loss | — |
| 11 | B | Factors for the RL agent predicted chunk by chunk with the leaky model, carrying hidden state across chunks | Factors inherit both leaks | Overlapping windows from a fresh state, ≥ `wup_dim` bars of past context — the condition the model was trained under | `tests/test_causality.py::test_factor_prediction_is_causal` |
| — | Q | A "train–valid gap > 0.5 ⇒ extra patience" rule | — | Removed (never triggers with MSE on [−1, 1] targets; not needed) | — |

The original leaky architecture remains available as `arch="enhanced_legacy"`, only for the leak demonstration in notebook 02.
