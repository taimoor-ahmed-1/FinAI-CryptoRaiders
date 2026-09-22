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

## Trading environment and agent — `5_erl_trainer`, `6_erl_evaluator`

| # | Sev | Original behaviour | Effect | Fix | Verified by |
|---|---|---|---|---|---|
| 12 | **B** | Training loop sends `action - 1` and `TradeSimulator.step` subtracts 1 again → position change ∈ {−2, −1, 0}. The evaluator sends `action + 1` → {0, +1, +2} | In training the agent can only sell/short or hold (a position can never be increased); in evaluation it can only buy or hold. Training and evaluation act in **different action spaces** | One mapping everywhere: action {0, 1, 2} → change {−1, 0, +1} | `tests/test_env.py` |
| 13 | **B** | The state is read with `env.get_state(env.step_is)`, the episode's **start** index, not the current bar `step_is + step_i` (trainer and evaluator) | Market features never change within an episode; the agent is blind to the market it trades | State read at the current bar | `tests/test_env.py::test_trade_executes_at_next_bar_and_state_is_current` |
| 14 | **B** | 1-minute episodes: `seq_len` 64 − 60 ignored steps, `// step_gap` 2 → **2 decisions per episode**, then a forced close. The evaluator also uses `max_step` = 2 | Nothing longer than ~4 minutes can be learned; the "backtest" is 2 steps from a random start | Training episodes of 4,320 bars (3 days); evaluation is one continuous pass over the whole split | — |
| 15 | **B** | Evaluator: `EVAL_DATA_SPLIT = "valid"`, but prices are read from `..._train_70.csv` (first match in its search list) | Validation factors are paired with **training-period prices** — the reported metrics describe no real period | Each split's prices, factors and signals come from the same prepared split; reported numbers are on **test**, with the checkpoint chosen on validation | `runner.py` |
| 16 | **B** | Only simulation 0's state is used; its action is broadcast to all 64 copies; only its reward is learned | 63/64 of the simulation is wasted and inconsistent | Vectorised: every copy acts on its own state and contributes transitions | — |
| 17 | R | `stop_loss_thresh = 1e-3` compared with a **dollar** price difference | Any position is closed at the first $0.001 tick against it — a hidden forced exit on almost every step | Relative trailing stop (`stop_loss=0.02` = 2 %), **off by default** | — |
| 18 | R | Slippage 7e-7, no fee | Effectively frictionless | Binance spot taker fee 0.10 % + 0.02 % slippage per side, on traded notional | `tests/test_env.py::test_round_trip_costs_at_flat_price` |
| 19 | R | Trades fill at the midpoint of the bar just observed | `midpoint` is the bar's (high+low)/2, only known after the bar | Fill at the next bar's midpoint (`exec_lag=1`) | `tests/test_env.py` |
| 20 | R | Position = 1 BTC against $1 M cash (~4–10 % exposure) | Not comparable with buy & hold | Position level = fraction of equity (fully long = all equity, sized net of the entry cost); buy & hold on the same basis | `tests/test_env.py::test_buy_and_hold_closed_form` |
| 21 | **L/B** | LLM signals min-max normalised over whichever split is loaded (the evaluator rescales on the evaluation split itself) | Test statistics enter the test state; the same raw value maps to different inputs in training and testing | Fixed scale of the 1–5 LLM score: (x − 1) / 4 | — |
| 22 | **B** | PPO samples actions at temperature 1.2 but recomputes log-probabilities at temperature 1 in the update | The probability ratio is wrong from the first epoch; the clipped objective is biased | Same distribution for sampling and update | — |
| 23 | B | Critic trained on **normalised** returns while GAE uses the un-normalised values | Critic targets and estimates are on different scales | Critic targets = GAE returns | — |
| 24 | B | The last value of every rollout is taken as 0; time-limit ends treated as terminal | Biased advantages at every boundary | Bootstrapped with V(s_T) | — |
| 25 | Q | Dropout 0.1 active during rollouts and updates | The policy that acted differs from the one updated | Dropout 0 (configurable) | — |
| 26 | Q | Orthogonal init with gain 0.01 on **every** actor layer | Activations shrink layer by layer | Gain √2 on hidden layers, 0.01 on the policy head only | — |
| 27 | — | Reward shaping: +0.001 per position change, +1 % of positive rewards | Pays the agent to trade, which works against transaction costs | Default reward is the plain change in net assets; the shaping remains available as `reward="legacy_shaped"` | — |
| 28 | — | State dimension and signal columns hard-coded (12) | Ablation impossible without editing code | `EnvConfig(state="A"/"B"/"C"/"D")` or explicit `signals=(...)` | `tests/test_env.py::test_state_dims` |

Unchanged on purpose: chronological 70/15/15 split (`int(n·0.70)`, `int(n·0.85)`), the label
definition (`seq_to_label`), the Alpha101 formulas, the RNN architecture and its hyperparameters
(as defaults), the PPO hyperparameters (as defaults), the discrete {decrease, keep, increase}
action space with shorting allowed, and 2 M training transitions.

## Additions

| Area | What |
|---|---|
| RNN architectures | LSTM, GRU, Transformer encoder (causal mask), TCN (dilated causal convolutions) next to the LSTM+GRU hybrid — `rnn/zoo.py` |
| RNN search | Optuna per architecture, validation MSE, SQLite-backed (resumable), final multi-seed runs, selection rule — `rnn/tune.py` |
| Agents | A2C, DQN, Double DQN, D3QN next to PPO — `agents/` |
| Agent search | Optuna per agent on validation Sharpe — `tune_rl.py` |
| Rewards | differential Sharpe, drawdown-penalised — `rewards_risk.py` |
| Evaluation | validation-based checkpointing, one exact test backtest, all brief §4.1 metrics net of costs — `runner.py`, `metrics.py` |
| Traceability | stable run ids, one `results.csv` row per run, weights + config + seed + commit per run, environment lock per notebook |
