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
