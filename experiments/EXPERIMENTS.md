# EXPERIMENTS — running log

What was tried, the configuration, the result, and **why the next thing was tried**.
Negative results stay in. Each entry names the notebook, branch and commit (printed in
the notebook's first cell and stored in every `results.csv` row) so it can be traced.

Template:

```
## E<n> · <title>                       <date> · <notebook> · <branch>@<commit>
Config:   what was run (only what differs from the defaults), seeds, budget
Result:   numbers (test, net of costs, mean ± std over seeds) + where the table/figure is
Reading:  what it means, what was surprising
Next:     what this motivates, and why
```

---

## E0 · Code audit and fixes                 2026-09 · (no run) · exp/02-leakage-fixes

Config: review of the six provided notebooks against brief §3.3.
Result: six look-ahead paths and a series of result-changing bugs; see `LEAKAGE_AUDIT.md`
and `CHANGES.md`. Tests added for each fixed leak; all pass.
Reading: earlier PoC numbers cannot be used as a reference point. Both the RNN
validation curve and the agent's backtest were affected.
Next: rebuild the baseline on the fixed pipeline (E1–E3) before any tuning.

## E1 · Data preparation                                   · 01_prepare_data
Config: default (70/15/15, train-fitted normalisation, one-day signal lag)
Result: split dates ___ ; tests ___
Next:

## E2 · RNN, default configuration (causal)                · 02_rnn_train
Config: `enhanced`, notebook hyperparameters, seed 0
Result: stop reason ___ at step ___; val MSE ___, IC ___; test MSE ___, IC ___
Leak demo (optional): enhanced ___ vs enhanced_legacy ___
Reading: does validation loss still fall monotonically?
Next:

## E3 · Baselines — Milestone 1                             · 03_baseline
Config: buy & hold; random × 5; PPO state A × 5, notebook hyperparameters, 2 M transitions
Result: (paste `report/milestone1_baseline.md`)
Reading:
Next:

## E4 · RNN tuning                                          · 04_rnn_tuning
Config: trials per architecture ___, trial budget ___, final seeds ___
Result: (paste `report/rnn_leaderboard.md`); selected ___
Reading:
Next:

## E5 · Ablation A–D                                        · 05_rl_ablation
Config: PPO, same hyperparameters for A–D, 5 seeds; factors from ___
Result: (paste `report/ablation.md`), incl. D − B (RQ3) and the same-day-signal check
Reading:
Next:

## E6 · Agent comparison                                    · 06_agent_comparison
Config: trials per agent ___, tuning budget ___; best hyperparameters in `runs/tuning/best_hp.json`
Result: (paste `report/agents.md`)
Reading:
Next:

## E7 · Reward comparison                                   · 07_reward_comparison
Config: agent ___ (best validation Sharpe in E6), state D, 5 seeds per reward
Result: (paste `report/rewards.md`); reward chosen on validation: ___
Reading:
Next:

## E8 · Final model                                         · 08_results
Configuration chosen on validation: ___ ; test result: ___

## E9 · Contest week: news signals in the ablation    2026-10-29 · 09_contest_news · exp/09-contest-news
Why: the tweet signal is a daily aggregate that mostly echoes past prices (signal check), so
E5 could not show an effect. The FinAI Contest 2025 Task 1 data has INTRADAY LLM signals: the
organisers' DeepSeek scores of 279 Bitcoin news articles (HF `FinRL_BTC_news_signals`), with
publication times, over the contest's 1-second order book (2021-04-07 → 04-13).
Data:   `rltrade/contest.py`. 1-minute bars built like the thesis's Binance bars; signals rebuilt
        from publication times - a bar sees only articles available before it ends, and an article
        waits for its Yahoo URL time when that is later (236/240 agree; 4 differ). The rebuilt
        signal matches the organisers' merged file on 97 % (sentiment) / 99 % (risk) of bars; every
        difference comes from those conservative choices. No look-ahead in the organisers' merge
        (bar values = last article at or before the bar, 98 %; next article: 42 %).
Config: same pipeline, only the data changes: Alpha101 (normalisation fitted on contest train) →
        the GRU selected in E4 applied unchanged (contest IC train/val/test +0.17/+0.19/+0.18) →
        DDQN with the E6 hyperparameters, not re-tuned. Contest split 70/15/15 (train 4 days, val and
        test ~20 h each). 12-hour episodes, 300k transitions, validation every 10k (exact pass),
        ≥ 5 validation trades. States A–D at 0.10 % + 0.02 %; A and D also at 0.02 % per trade.
        10 seeds each; buy & hold and cash at both cost levels.
Caveat: 20-hour test window - seeds share one market path, p-values indicative only.
Result: test return net of costs, mean (median) of 10 seeds; buy & hold -0.44 % / -0.34 %.
          0.12 %:  A -2.76 (-1.09)   B -2.71 (-0.87)   C -4.06 (-0.54)   D -0.62 (-0.81)
                   before costs: A +0.16  B -0.02  C +0.11  D -0.11
                   Welch on test Sharpe: B vs A p = 0.88, C vs A 0.68, D vs A 0.25, D vs B 0.22
          0.02 %:  A -0.11 (-0.07)   D -0.51 (-0.30);  D vs A p = 0.11 (Sharpe), 0.13 (return)
        Full tables and figures: `contest/report/` (local, not committed - built from the
        confidential contest data).
Reading: the news signals do not improve the agent. Before costs every state ends near zero
        (-0.1 % to +0.2 %), so no state found an edge on the test window. The spread of the
        net means at 0.12 % comes from 5 runs (A s7/s8, B s7, C s6/s7) whose only eligible checkpoint
        (>= 5 validation trades) was an early, high-turnover one (134-163 trades, val Sharpe down
        to -214): ~150 trades x 0.12 % = ~18 % in costs. Later checkpoints traded < 5 times and
        were not eligible. D had no such run, which is why its mean looks better; its median
        (-0.81 %) is between A's and B's. At 0.02 % the agent with both signals is, if anything,
        worse than price only (p = 0.11). Consistent with E5: no evidence that the LLM signals
        help, here with intraday, causally timed signals as well.
Next:   the minimum-trades rule can force a bad checkpoint when the agent converges to
        inactivity; a fallback to "stay flat" (or reporting both) would separate that artefact from
        the signal question.
