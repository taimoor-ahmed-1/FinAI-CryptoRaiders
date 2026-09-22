# Experiments — hybrid RL + LLM risk-aware BTC trading

Experiment code for the thesis *A Hybrid Reinforcement Learning and Large Language
Model Framework for Risk-Aware Cryptocurrency Trading*. Everything here runs on
Google Colab; data and results live on Google Drive. **The dataset is never committed.**

Start with [`LEAKAGE_AUDIT.md`](LEAKAGE_AUDIT.md) and [`CHANGES.md`](CHANGES.md): the provided
pipeline leaked future information in six ways and had several bugs that changed its
results. This code fixes them. Earlier results should not be compared with results from it.

## Branches

Each branch builds on the previous one and adds one step of the work. `master` is untouched.

| Branch | Adds | Notebook(s) | Brief |
|---|---|---|---|
| `exp/01-setup` | the provided notebooks, unmodified (`original/`), docs, pinned requirements | `original/1_…6_` | §3.1 |
| `exp/02-leakage-fixes` | `rltrade` package: data preparation, Alpha101 (vectorised, proven identical), per-split labels, causal RNNs, tests | `01_prepare_data`, `02_rnn_train` | §3.3 gate |
| `exp/03-baseline` | trading environment, metrics, PPO, buy & hold, random, runner — **Milestone 1** | `03_baseline` | §3.2 |
| `exp/04-rnn-tuning` | Optuna search over LSTM / GRU / hybrid / Transformer / TCN | `04_rnn_tuning` | §3.3 |
| `exp/05-rl-ablation` | ablation A–D (same agent, same hyperparameters) + sentiment-leak check | `05_rl_ablation` | §3.4 a |
| `exp/06-agent-comparison` | A2C, DQN, Double DQN, D3QN; per-agent Optuna search | `06_agent_comparison` | §3.4 b |
| `exp/07-reward-comparison` | differential-Sharpe and drawdown-penalised rewards | `07_reward_comparison` | §3.4 d |
| `exp/08-results` | final tables, figures, model index — **Milestone 2** | `08_results` | §3.5, §4 |

## Running on Colab

1. Upload `BTC_1min_with_sentiment_risk_train.csv.gz` to **`MyDrive/thesis_rl/data/`**.
2. Open a notebook straight from GitHub, e.g.
   `https://colab.research.google.com/github/taimoor-ahmed-1/FinAI-CryptoRaiders/blob/exp/08-results/experiments/notebooks/01_prepare_data.ipynb`
   (swap in any notebook name). Runtime → Change runtime type → **T4 GPU** (01 and 08 are fine on CPU).
3. **Runtime → Run all.** The first cell mounts Drive, checks out *only* `experiments/` of the
   notebook's branch (the repo's data folders are never downloaded), installs `optuna`, and records
   the exact environment (`pip freeze`, Python, CUDA, GPU) to `thesis_rl/report/environment/`.

Run in this order:

| # | Notebook | Needs | Rough T4 time |
|---|---|---|---|
| 01 | prepare data | dataset on Drive | ~5 min |
| 02 | RNN, default configuration | 01 | 30–90 min |
| 03 | baselines (Milestone 1) | 02 | ~1–2 h (5 PPO seeds) |
| 04 | RNN tuning | 01 | the longest step; set trials and time caps in the notebook |
| 05 | ablation A–D | 02 (or 04) | ~4–8 h (25 PPO runs incl. the leak check) |
| 06 | agent comparison | 05's factors | tuning + 25 runs; spread over sessions |
| 07 | reward comparison | 06 | ~15 runs |
| 08 | results | any of the above | minutes |

**Disconnects lose nothing.** Every finished run is skipped when a notebook is re-run; Optuna
studies are stored in SQLite on Drive and resume. Just run the notebook again.

## Drive layout

```
thesis_rl/
  data/BTC_1min_with_sentiment_risk_train.csv.gz     input
  prepared/            factors, labels, market arrays (01)
  rnn/default/         default RNN + its factors (02)
  rnn/tuning/          studies, trials_*.csv, final/<arch>_s<seed>/, selected.json (04)
  runs/results.csv     one row per run: every seed, every metric (03, 05-07)
  runs/models/<experiment>/<agent>_<state>_<reward>_s<seed>/
                       model.pt, config.json, metrics.json, history.json, test_backtest.npz
  runs/tuning/         RL studies, trials_*.csv, best_hp.json (06)
  report/              markdown tables, figures, model_index.csv, environment/
```

## Conventions

- **Split:** chronological 70/15/15, as the provided splitter. Train 2024-01 → 2025-09, validation → 2026-01, **test 2026-01 → 2026-06**.
- **Selection:** RNN checkpoints and architectures on validation MSE; RL checkpoints, hyperparameters, agent and reward on validation Sharpe. **Test is evaluated once per run and never used to choose anything.**
- **Seeds:** 5 per reported configuration (0–4); tuning trials use seeds ≥ 1000 and are not reported. Seeds set for Python, NumPy, PyTorch and CUDA.
- **Costs:** 0.10 % Binance spot taker fee + 0.02 % slippage per side on traded notional; fills at the next bar's midpoint. Every reported metric is net of costs.
- **Metrics** (`rltrade/metrics.py`): cumulative and annualised return, Sharpe and Sortino (annualised, 24/7), max drawdown, RoMaD, win rate over round trips, trades, turnover, time in market, costs paid.
- **State configurations:** A price-only (10), B + sentiment (11), C + risk (11), D + both (12). LLM signals are the previous completed day's aggregate.
- **Actions:** decrease / keep / increase the position level; levels −1, 0, +1 (short, flat, long), as in the provided environment.

## Code map

```
rltrade/
  prepare.py        split, Alpha101, labels, market arrays (+ one-day signal lag)
  alpha101.py       the 101 factors; vectorised helpers
  labels.py         multi-horizon trend labels, per split
  rnn/              models.py (provided), zoo.py (LSTM/GRU/Transformer/TCN), train.py, tune.py
  env.py            vectorised trading environment, backtester
  rewards.py        net_asset_change, legacy_shaped (+ rewards_risk.py: diff_sharpe, drawdown_penalized)
  agents/           ppo.py, a2c.py, dqn.py (dqn / ddqn / d3qn), nets.py
  baselines.py      buy & hold, random
  runner.py         train → validation checkpointing → one test backtest → results.csv
  tune_rl.py        per-agent Optuna search
  metrics.py, results.py, seeding.py, paths.py
tests/              causality, Alpha101 equivalence, signal lag, environment accounting, rewards
notebooks/          01-08 (Colab)
original/           the provided notebooks, unmodified (reproduction, brief §3.1)
```

Local runs: `pip install -r requirements.txt`, then run the tests (`python tests/<file>.py`) or open
the notebooks from `notebooks/` (they fall back to `../../thesis_rl_outputs`, or `$THESIS_RL_ROOT`).

## Reproducing the provided pipeline (brief §3.1)

`original/` holds the six provided notebooks exactly as delivered. They expect the
`btc_rl_task` folder layout on Drive: upload the notebooks and `data/` there, run
`1_colab_setup` once, then 2 → 6. See `CHANGES.md` for why their outputs are not
comparable to the results produced here.
