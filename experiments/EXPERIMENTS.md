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

## E10 · Contest starter kit, bugs fixed (price only)   2026-10-31 · 10_starter_kit_baseline · exp/10-starter-kit
Why: the organisers' own Task 1 pipeline is the natural baseline for the contest data. As shipped
        (`break_step = 32`, run locally on a held-out 70/15/15 split) it never trades: return 0,
        drawdown 0, Sharpe undefined over all 14 test episodes; buy & hold 1 BTC -0.005 %.
Code:   `starter_kit/` - commit 428db91 is the kit unmodified (MIT); every change is marked `[fix]`
        and listed in `starter_kit/README.md`. Main ones: break_step 32 -> 32e4 (the kit's TODO);
        factors were 900 rows (15 min) stale against prices; the 4 ensemble members shared one seed
        (3 were identical) and one was overwritten on save; exploration used the target network;
        the checkpoint was scored on training data and then discarded; the stop-loss was an
        absolute $0.001 on a $60k price; the evaluator priced trades at the episode counter.
Config: Alpha101 over the whole 1-second file (causal), scaled with train quantiles -> the kit's
        LSTM+GRU (fit on the first 60 % of train, early-stopped on the rest) -> D3QN, DoubleDQN x2,
        TwinD3QN, kit hyperparameters, 320k steps each, 4096 envs (fewer if the GPU is small),
        best checkpoint on all 14 validation episodes -> majority vote, test once on its 14
        consecutive 79-minute episodes. Slippage 7e-7 as shipped; 1 BTC position on $1M.
Run:    Colab A100-40GB, 4096 envs (as shipped), 2026-10-31 18:28 -> 22:54, ~63-74 min per member.
        Best validation return per 79-min episode: D3QN $578 (step 322k, the last one - still
        rising), DoubleDQN $108 (228k), DoubleDQN $130 (294k), TwinD3QN $520 (275k). All four sat
        at "hold" ($0) for the first ~100-150k steps; the 3 that were identical as shipped now differ.
Result: chained 79-min episodes, 14 each, $1M, 1 BTC position, slippage 7e-7 (simulator account):
                      return    Sharpe/step   max DD    trades   episodes > 0
          val         +0.773 %     0.042      -0.036 %   3,334
          test        +0.757 %     0.037      -0.093 %   3,514     12 / 14
          B&H 1 BTC   -0.021 %    -0.001      -0.137 %             7 / 14   (test)
          B&H full    -0.338 %    -0.001      -2.26 %              7 / 14   (test)
        Full tables and figures: Drive `starter_kit/runs/shipped/` (not committed).
Reading: with the bugs fixed the kit trades actively and earns a small, steady gross edge that
        holds out of sample (test = validation, 12 of 14 episodes positive) while the market fell.
        But it is ~$2.2 per round of trading on ~$57k notional (~0.4 bp), executed at the mid with
        ~zero cost; the simulator charges no spread or fee. At any realistic cost (taker fee
        ~4-10 bp; the thesis uses 12 bp) ~3,500 trades cost far more than the edge (at 12 bp ~ -24 %).
        So the shipped baseline is a frictionless result: it shows the factors carry very
        short-horizon information, not a tradable strategy.
At 0.12 % (slippage 1.2e-3, `runs/slippage_0.12pct`, RUN_REALISTIC; 2026-11-01 02:23 -> 06:55, everything
        else identical): all 4 members kept a never-trade checkpoint - best validation return $0.00 at
        every evaluation over 320k steps each. Test: 0 trades, return 0, drawdown 0, Sharpe undefined
        (buy & hold 1 BTC -0.021 %). With realistic costs the kit's agents learn that no trade pays: the
        frictionless edge (~0.4 bp per trade) is far below the 12 bp cost, and they find nothing larger.
Next:   a longer budget at 7e-7 (two members peaked at the last checks) would only enlarge a
        frictionless result; the cost result is the thesis-relevant baseline.

## E11 · Contest starter kit with the DeepSeek news signals   2026-11-01 · 11_starter_kit_news · exp/11-kit-news
Why: E10 gives the kit's own pipeline, fixed, on price only. The kit was built to add the organisers'
        DeepSeek sentiment / risk scores of the week's news to the state; does that help, with the
        scores mapped onto the 1-second data properly?
Mapping: `starter_kit/news_signals.py`. The organisers held the latest article's score until the next
        one (E9 did the same, causally). Here: the 232 of 279 articles that mention bitcoin / crypto;
        usable from max(publication, Yahoo URL time) + 60 s; per second a decayed, confidence-weighted
        mean of (score - train mean), half-life 1 h, with a neutral prior weight of 1, so it fades to 0
        between articles and several agreeing articles count more than one; scaled by the train 99th
        percentile of |s|. Fixed in advance. Brute-force causality check: 324 rows, 0 mismatches.
Config: identical to E10 (same prepared data and RNN factors, members, seeds, 320k steps, 4096 envs,
        slippage 7e-7, validation selection, 14 test episodes) + 2 state inputs. Compared with
        `runs/shipped`. Runs on a second Colab runtime alongside E10's 0.12 % run.
Caveat: one ensemble against one ensemble on a 20-hour test window: a difference shows what the signals
        did here, not a significant effect. RL training is chaotic, so the extra inputs alone change every
        trajectory.
Run:    Colab A100-40GB, 4096 envs, 2026-11-01 02:41 -> 07:26 (parallel to E10's 0.12 % run).
        Best validation $ per 79-min episode, news vs price only: D3QN 110 vs 578, DoubleDQN 130 vs 108,
        DoubleDQN 75 vs 130, TwinD3QN 125 vs 520. All started trading at ~120-180k steps as before; the
        two late jumps of the price-only run (D3QN, TwinD3QN after ~275k) did not happen.
Result: chained 79-min episodes, 14 each, $1M, 1 BTC, slippage 7e-7 (simulator account):
                           return    Sharpe/step   max DD    trades   episodes > 0
          val  price only  +0.773 %     0.042      -0.036 %   3,334    14 / 14
               with news   +0.146 %     0.011      -0.054 %     238     9 / 14
          test price only  +0.757 %     0.037      -0.093 %   3,514    12 / 14
               with news   +0.045 %     0.003      -0.083 %     278     6 / 14   (median episode -0.003 %)
               B&H 1 BTC   -0.021 %    -0.001      -0.137 %             7 / 14
        Drive `starter_kit/runs/news_decay/` (comparison.csv, 11_starter_kit_news.png).
Reading: no evidence that the news signals help; in this run the ensemble with them trades ~12x less and
        earns about what buy & hold loses, while the price-only ensemble keeps its small frictionless edge.
        Validation ranked the two the same way, so this is not a test-only accident, but it is one
        ensemble each: much of the gap is two price-only members whose late jump (to $520-578) did not
        recur, which may be training luck as much as the inputs. Consistent with E5 and E9 (no benefit
        from the LLM signals at daily, 1-minute or 1-second resolution). At realistic cost both would be
        expected to stay flat (E10 at 0.12 %: 0 trades), so a 0.12 % news run was not done.

## E12 · Interval-valued, multi-scale news signals (after Tian et al. 2026)   2026-11-02 · rltrade.contest_intervals · exp/12-paper-intervals
Why: Tian, Yang, Wei & Wang (2026, Pacific-Basin Finance Journal 103352) forecast Bitcoin with interval-valued
        series (weekly [q25, q75] of news sentiment), frequency-specific predictors (sentiment for the fast part,
        attention for the medium part) and trade only when the point forecast lies inside the forecast interval.
        Do these ideas make the news signals useful to our agent, where E9's point scores were not?
Pre-check, the paper's own scorer (not run): VADER on title + first paragraph, rolling [q25, q75] over the last
        28 articles (the paper's articles per week) - `starter_kit/vader_signals.py`. Causal (300 rows, 0
        mismatches), but VADER rates 20 % of the articles exactly 0 (general lexicon: "Bitcoin surges to record
        high" = 0), agrees weakly with DeepSeek (Spearman 0.14), and the interval spans ~14 h, changing 4 times in
        validation. No IC significant (p >= 0.06 at 1 min - 4 h). Not run; DeepSeek scores used instead.
Data:   `rltrade/contest_intervals.py`; E9's bars, timing (articles available before the bar ends) and factors,
        the 232 crypto-relevant articles (as E11). Per bar, over the articles in a trailing 6 h and 24 h window
        (>= 3 articles, else 0): centre (q25 + q75) / 2 and width q75 - q25 of the DeepSeek sentiment and risk
        (8 columns) + news intensity log((n_6h + 1) / (n_24h / 4 + 1)) for attention (Google Trends not used).
        Scaled with train mean / train p99 of |x - mean|, clipped to [-1, 1]. NA-MEMD not used (decomposes the
        whole sample - look-ahead). Causality check: 300 bars x 9 columns, 0 mismatches.
        6 h interval present on 91 % of bars; sentiment centre changes 117 / 10 / 23 times in train / val / test.
Gate:   state F - the environment blocks opening or enlarging a position while the 6 h sentiment interval is
        wider than its train median (0.5); reducing / closing always allowed; no interval = open. Open on
        55 / 81 / 50 % of train / val / test bars. Random-policy check: 0 of 190 entries with the gate shut (E: 147).
Pre-check IC (Spearman with the future return, p from 200 circular shifts): 6 h sentiment centre at 4 h
        -0.29 (p 0.01) train, -0.43 (0.01) val - contrarian, as DeepSeek sentiment in the VADER pre-check;
        no other column consistent on both splits.
Config: identical to E9 (DDQN, E6 hyperparameters, 12-h episodes, 300k transitions, validation every 10k,
        >= 5 validation trades, 10 seeds) - states E (A + 9 columns, dim 19) and F (E + gate), at 0.12 % and
        0.02 %. A, D, buy & hold and cash are E9's runs (same code path; the gate is off for them).
Run:    local CPU, 4 groups in parallel (2 threads each), 2026-11-02 18:23 -> 11-03 02:08, ~49 min per run.
        Code uncommitted at run time (runs record cb8adf0-dirty).
Result: test, mean (median) of 10 seeds; Sharpe annualised; max drawdown; return before costs (mean).
          0.12 %:  A  -2.76 (-1.09) %  Sharpe -31.9 (-9.9)   MDD 3.89 (2.24) %  gross +0.17 %  1/10 profitable
                   D  -0.62 (-0.81)          -5.5 (-7.2)         2.19 (2.13)       -0.11       2/10
                   E  -2.44 (-1.87)         -26.2 (-19.2)        3.81 (3.65)       +0.38       1/10
                   F  -0.79 (-0.56)          -9.0 (-5.7)         2.50 (2.33)       +1.15       3/10
                   buy & hold -0.44 %, Sharpe -3.76, MDD 2.12 %
                   Welch: E vs A p 0.81 (Sharpe), F vs A 0.33, E vs D 0.025 (E worse), F vs E 0.077
          0.02 %:  A  -0.11 (-0.07) %  Sharpe -0.6 (-0.4)    MDD 1.93 (1.96) %  gross +0.21 %  4/10
                   D  -0.51 (-0.30)          -4.7 (-2.6)         2.14 (2.01)       -0.18       1/10
                   E  +0.44 (+1.06)          +3.4 (+11.2)        2.23 (1.54)       +1.67       7/10
                   F  +0.64 (+0.70)          +7.1 (+9.4)         1.34 (1.37)       +1.37       7/10
                   buy & hold -0.34 %, Sharpe -2.86, MDD 2.12 %
                   Welch: E vs A p 0.64 (Sharpe) / 0.47 (return), F vs A 0.12 / 0.083, E vs D 0.35, F vs E 0.70
        Local `contest/report_e12/` (summary.csv, tests.csv, all_runs.csv, precheck_ic.csv; not committed).
Reading: at 0.02 % the interval states are the first news states in this project with a positive mean and
        median test return (E +0.44 %, F +0.64 %, 7 of 10 seeds profitable each, vs A -0.11 %, 4/10, and D
        -0.51 %, 1/10), and the gate lowers the drawdown (1.34 % vs 1.93-2.23 %). None of it is significant
        (best: F vs A, p = 0.08 on return), and it is one 20-hour test window shared by all seeds. At 0.12 % no
        state is profitable: E trades more (median 21) and loses to costs, worse than D (p = 0.025); F earns the
        largest gross return (+1.15 %) but costs take it, and is not distinguishable from A or D. So the paper's
        interval + gate idea is the most promising news variant so far at low cost and does not survive realistic
        costs - consistent with E9 / E11 that the signals carry at most a small edge.
Next:   confirm on more data before claiming anything (the contest week is the only intraday news set);
        the gate alone on state A (price-only + gate) would separate the gate from the interval inputs.

## E13 · The paper's daily interval-decomposition features on our data   2026-11-05 · 13_paper_daily · exp/12-paper-intervals
Why:    E12 borrowed Tian et al.'s ideas (intervals, multi-scale, gate) on the contest week. E13 rebuilds their
        dataset itself on our 2024-26 data and gives it to the E5 agent: the same seven interval series, the same
        aggregation, NA-MEMD and sample-entropy grouping, then features per frequency band.
Data:   `rltrade/paper_daily.py`, 888 UTC days 2024-01-01 -> 2026-06-06, daily [min, max] intervals of BTC
        (Binance 1-min midpoints), gold / WTI / DXY / S&P 500 (Yahoo GC=F, CL=F, DX-Y.NYB, ^GSPC, Low / High,
        forward-filled over closed days; the paper uses Investing.com), Google Trends "Bitcoin" (daily windows
        stitched on overlaps; the value is lagged one day because it is published late; [min, max] over the
        trailing 7 days), tweet sentiment [q25, q75], with VADER (the paper's scorer) and with DeepSeek. Data
        stays local (`data/paper_e13/`, `thesis_rl/prepared_e13/`), never committed.
Deviations from the paper (each needed on our data; see the module docstring):
        - daily, not weekly: the agent trades 1-minute bars, and weekly values would change 19 times in the test;
        - causal NA-MEMD: recomputed every day on the days up to then (expanding, >= 120 days; 64 directions,
          10 sifts, 2 noise channels at 0.2 std); the paper decomposes the whole sample (look-ahead);
        - sentiment ties: VADER ignores tweets scored exactly 0 (25 %; else q25 = 0 on 83 % of days); DeepSeek's
          1-5 integers use a continuity-corrected quantile (plain quartiles are [3, 4] on 95 % of days);
        - empty high band: daily tweet sentiment is so noisy (sample entropy ~1.8-2.1 vs <= 1.45 for every IMF)
          that the two-step rule puts all of it in low on every day, removing the paper's sentiment channel;
          the same happens to BTC on 98-194 of 769 days. When the high band is empty, IMF 0 (the fastest) becomes
          it. Literal-rule build kept locally in `data/paper_e13/literal_rule/` for reference;
        - event dummies (halving, Trump cycle) built but not in the state: chosen with hindsight, 0 on all
          validation and test days.
Features (12 per scorer, train-mean / train-p99 scaled, clipped to [-1, 1], 0 before day 120): BTC price vs its low
        band (trend gap), medium-band centre and high-band centre / width relative to price; 5-day change of the
        low band of gold, WTI, DXY, S&P 500; Google Trends medium-band centre / width; sentiment high-band centre
        / width. Lagged one UTC day onto the bars (bars on D see the features of D-1): 9,000 sampled bars, 0
        mismatches. Gate (state H): open unless D-1's raw sentiment interval is wider than its train median; open
        on 50 % of train days, 35 % (VADER) / 56 % (DeepSeek) of validation days.
Pre-check (Spearman IC of D-1 features with day D's return and |return|, p from 500 circular shifts >= 30 days,
        train and validation only): 7 of 96 tests at p < 0.05 (chance ~5), none with the same sign and p < 0.1 on
        both splits. Strongest: BTC trend gap vs |return| (DeepSeek build) -0.14, p 0.002 on train, not on
        validation; DeepSeek sentiment high band vs return +0.08 / +0.11 (p 0.10 / 0.22), its width +0.09 / +0.11
        (p 0.03 / 0.21) - same sign on both splits, weak; VADER sentiment nothing (|IC| <= 0.07).
Config: E5's agent unchanged (DDQN, E6 hyperparameters `best_hp_mt5`, E4 GRU factors gru_s1, 2M transitions,
        validation every 100k, >= 5 validation trades, 5 seeds), states Gd / Hd (DeepSeek) and Gv / Hv (VADER),
        dim 22; DeepSeek states run first, VADER after. A and D are E5's runs.
Result: pending.
