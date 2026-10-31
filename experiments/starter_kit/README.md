# FinAI Contest 2025 Task 1 starter kit — bugs fixed (experiment 10)

The organisers' `Task_1_FinRL_DeepSeek_Crypto_Trading` code (SecureFinAI Lab, MIT — see `LICENSE`,
original README in `KIT_README.md`), used as a baseline on the contest's 1-second order-book data,
**price only** (the DeepSeek sentiment/risk inputs are removed). Commit `428db91` is the kit unmodified, so

```bash
git diff 428db91 -- experiments/starter_kit
```

shows every change. Each is marked `[fix]` (or `[baseline]` for the price-only state) in the code.

Run it with `notebooks/10_starter_kit_baseline.ipynb`, or locally:

```bash
python kit_pipeline.py prepare  --csv BTC_1sec.csv --data prepared   # CPU, ~3 h (Alpha101)
python kit_pipeline.py rnn      --data prepared
python kit_pipeline.py ensemble --data prepared --run runs/shipped     # GPU
python kit_pipeline.py evaluate --data prepared --run runs/shipped --split test
```

The data (`BTC_1sec.csv` and everything derived from it) is confidential and is never committed.

## As shipped

Run locally with the kit's own scripts on a held-out 70/15/15 split (`break_step = 32`): the ensemble never
trades. Every member picks "hold" on every test state; return 0, drawdown 0, Sharpe undefined over the 14 test
episodes; buy & hold 1 BTC −0.005 %, −0.12 % drawdown.

## Fixes

| file | as shipped | fixed |
|---|---|---|
| `task1_ensemble.py` | `break_step = int(32)  # TODO reset to 32e4` | 320,000 (argument) |
| `trade_simulator.py`, `seq_run.py` | the factor array is 900 rows shorter than the prices (the longest label horizon) and is aligned with the **rear** of the prices: at second t the agent sees the RNN forecast made at t − 900 s, whose horizon has already passed | one causal prediction pass over all rows; row t's factors sit next to row t's price |
| `task1_ensemble.py` | `random_seed = gpu_id` for every member, and `Config.cwd` set only once, so all members share a seed and a training folder: DoubleDQN ×2 and TwinD3QN (the same `QNetTwin` network) train into identical weights | seed + member index, a folder per member |
| `task1_ensemble.py`, `task1_eval.py` | members saved by class name, so the 2nd DoubleDQN overwrites the 1st; the evaluator loads 3 classes | `<idx>_<Agent>/` per member; all 4 vote |
| `erl_agent.py` | exploration with `act_target` (soft update τ = 2e-6) and its default ε = 0.125 (`# TODO check`) | the online network, ε = `explore_rate` (0.005) |
| `erl_agent.py` | `get_cumulative_rewards` bootstraps with `argmax` (an action index) instead of the max Q value | `max` |
| `task1_ensemble.py` | the training evaluator scores 3 random windows of the **training** file (`Ensemble.__init__` sets it to 1 env in place), and the best checkpoint is never used: the final weights are saved | every consecutive episode of the validation split; the best-validation actor is kept (`selection.json`) |
| `trade_simulator.py` | stop-loss an absolute $0.001 (train) / $0.0001 (eval) on a ~$60,000 price — any adverse tick closes the position, and train and eval differ | 0.1 % of the price (the kit's commented-out `mid_price * stop_loss_rate`), same in both |
| `task1_eval.py` | bookkeeping priced at `price_ary[step_i]` (the episode counter, not the row) | the traded row, read before `step()` (the env resets inside the last step) |
| `task1_eval.py` | one random 79-minute window; `dataset_path` never read | every consecutive 79-minute episode of the split, chained; the simulator's account is the headline |
| `seq_run.py` | the RNN's "validation" loss is sampled from its training rows; the last network is kept | held-out rows (last 40 % of train); best network |
| `seq_data.py` | each file's factors scaled with its own quantiles (test scaled with its own future) | train-split quantiles for all rows; constant column → 0 without NaN |
| `seq_data.py`, `seq_run.py` | per-file conversion; two diverging `ConfigData` classes | the whole file converted once (Alpha101 only looks back), splits as row ranges; one `ConfigData` |
| `task1_ensemble.py` | members saved only after all are trained | each saved when done; a re-run resumes |
| `erl_agent.py`, `metrics.py` | `torch.load` fails on torch ≥ 2.6; `empyrical` 0.5.5 does not install on Python 3.12 | `weights_only=False` (own checkpoints); `empyrical_compat.py` (the two functions, as in 0.5.5) |

## Kept as shipped

Slippage 7e-7 per unit traded, position −1/0/+1 BTC on $1M, max holding 1 h, 2-second steps, 79-minute
episodes, learning rate 2e-6, γ 0.995, soft update 2e-6, ε 0.005, network (128, 128, 128), replay buffer
75,840 steps × envs, 4096 parallel envs, majority vote (ties: the first member's action), Sharpe = mean / std of
per-step returns (not annualised). `kit_pipeline.py` halves the number of envs when the replay buffer (~24 GB
at 4096) does not fit the GPU; the value used is in `runs/<run>/config.json`.

## Experiment 11: with the DeepSeek news signals

`news_signals.py` maps the organisers' DeepSeek scores of the week's news (HF `SecureFinAI-Lab/FinRL_BTC_news_signals`,
279 articles) onto every second; `--signals` adds them to the state (2 + 8 + 2, as the kit had). Everything
else is experiment 10, so `runs/shipped` is the price-only comparison. Run it with
`notebooks/11_starter_kit_news.ipynb`, or

```bash
python kit_pipeline.py news     --csv BTC_1sec.csv --news news_train.csv --data prepared
python kit_pipeline.py ensemble --data prepared --run runs/news_decay --signals
python kit_pipeline.py evaluate --data prepared --run runs/news_decay --split test
```

| | kit (organisers' merged csv) | `news_signals.py` |
|---|---|---|
| articles | all | the 232 that mention bitcoin / crypto in the title or the first 600 characters |
| usable from | publication time | max(publication, Yahoo URL time) + 60 s, causal |
| value at second t | latest article's score, held | sum w_i (score_i - centre) / (1 + sum w_i), w_i = confidence_i * 2^(-age_i / 1 h) |
| no recent news | last score held for hours | fades to 0 = the train-period mean score |
| scale | 1-5 | / the train 99th percentile of \|s\|, clipped to [-1, 1] |

The settings (half-life, latency, prior, keyword list) were fixed before any training; none was tuned.
