# Overview - how the experiments were done, what each figure shows, data constraints

A short companion to `README.md` (which has the full tables). Read this first.

## 1. How we went about it

1. **Built the data** (2024-01-01 to 2026-06-06). We scraped about 278k Bitcoin tweets, and an LLM
   (DeepSeek-V3) scored each one for *sentiment* and *risk* on 1-5 scales; we averaged the scores per day.
   We rebuilt 1-minute BTC market data from Binance's public archive, because the contest file only
   covers 2021.
2. **Audited the provided pipeline before using it.** It leaked future information: same-day sentiment
   on every bar of that day, an unmasked attention layer, a carried-over hidden state, and per-split
   normalisation. It also had bugs that change results: mismatched training and evaluation actions, a
   frozen state, 2-step episodes, and almost zero trading costs. All of these were fixed and tested. Old
   proof-of-concept numbers are therefore not comparable with these results.
3. **Fixed the evaluation protocol before running anything:**
   - chronological 70/15/15 split;
   - signals lagged by one day;
   - realistic costs (0.10 % fee + 0.02 % slippage per side), orders filled at the next bar;
   - every choice made on **validation**, and the **test** period used once;
   - 5 seeds per configuration.
4. **Factor model.** The 101 price "alphas" are compressed into 8 factors by a causal recurrent network.
   We compared five architectures with Optuna, and GRU was selected (figure 03).
5. **Algorithms.** We tuned PPO, A2C, DQN, Double DQN and Dueling DDQN with Optuna (15 trials each),
   then trained each with 5 seeds (figure 04). Early runs showed that in a falling market, "do nothing"
   or "short once and hold" wins validation, so a checkpoint must now trade at least 5 times on
   validation to be selected.
6. **Main question: do the LLM signals help?** The best algorithm (Double DQN) is retrained with four
   input versions: A prices only, B + sentiment, C + risk, D + both (figure 06).
7. **Rewards.** Three reward functions are compared for that agent (figure 07).
8. **Final model.** We picked the configuration with the best validation Sharpe and reported it on test,
   per seed (figures 08-09).

Runs are reproducible: re-running a configuration with the same seed gives identical results.

## 2. What each figure shows

| # | Figure | What it shows |
|---|---|---|
| 01 | Price and splits | BTC price over the whole period, shaded by train / validation / test, with buy-and-hold return per split. Training rises, validation and test fall. |
| 02 | LLM signals | Daily sentiment, risk and tweet count (thin line = daily, thick = 30-day average). These are the inputs whose value figure 06 tests. |
| 03 | Factor models | Validation error and test predictive power (IC) of each architecture; dots = seeds. Close results, GRU chosen. |
| 04 | Algorithms | Test return and Sharpe of 5 RL algorithms; bar = mean of 5 seeds, dot = one seed, dashed = buy & hold. |
| 05 | Equity curves | How each algorithm's money evolves over the test period, against buy & hold and cash. |
| 06 | **LLM-signal ablation** | Same agent with inputs A/B/C/D, with Welch-test p-values. The main research result. |
| 07 | Rewards | The same agent trained with three different reward functions. |
| 08 | Final model, val vs test | Each seed's validation Sharpe (blue) next to its test Sharpe (orange): good validation does not carry over. |
| 09 | Final model equity | The final model's 5 seeds traded separately over the test period. |
| 10 | Trading activity | Number of trades per seed. The shaded zone (5 or fewer) marks near-static "enter once and hold" policies. |
| 11 | Overview | Mean and best-to-worst range of test Sharpe for every configuration on one axis. |

How to read the bar charts: the spread of the dots usually exceeds the difference between bars. With
only 5 seeds, differences are only meaningful where the p-value is below 0.05, and none is.

## 3. Data constraints

- **Tweets are a sample, not the full stream.** Collection used time-sliced sampling under a per-tweet
  API cost, about 300 tweets a day. The tweet count in figure 02 therefore reflects the sampling, not
  real market attention.
- **LLM scores are not validated against human labels.** They follow known market regimes (more
  bullish in the late-2024 rally, more bearish in the 2026 sell-off), but their accuracy per tweet is
  unmeasured.
- **Signals are daily; decisions are every 5 minutes.** A daily signal is constant through each day,
  and it is lagged one day to avoid look-ahead, so it can carry little intraday information.
- **Order-book depth is coarse:** 5 levels at fixed ±1-5 % from the mid price, sampled about every
  30 s, from the futures market. The prices are spot. The "spread" feature is the bar's high-low range,
  not a quoted bid-ask spread (Binance stopped publishing that data in early 2024).
- **One test window, in a falling market.** The test period (Jan-Jun 2026) loses about 32 % for buy and
  hold, and validation is also falling. A single short position can therefore look skilful. Results may
  differ in rising or sideways markets.
- **Regime shift between splits.** Training is a strong bull market (+174 %), while validation and
  test are bear markets.
- **Compute budget** (Colab T4): 15 tuning trials per algorithm, 2M training steps and 5 seeds per
  configuration. More seeds would narrow the uncertainty.
- **The dataset is unpublished.** The tweets and scores are not redistributed; only code and results
  are shared.

Left out of these results on purpose: superseded first attempts, and the runs that deliberately used
leaky same-day signals. They remain on Drive under `runs/` and `report/`.
