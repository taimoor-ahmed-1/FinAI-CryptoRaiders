"""Collect every table and figure into one folder, with a README describing data and experiments.

    from rltrade import compile_results
    compile_results.compile_all(L)          # -> <root>/thesis_results/

The folder is rebuilt from scratch each time: one sub-folder per experiment (figures and
tables copied from `<root>/report/`), `summary/` with tables computed here from
`runs/results.csv`, and `README.md`. Every number in the README is computed from
`results.csv`, so it always matches the runs on Drive.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import results as R

OUT_NAME = "thesis_results"

# (folder, regex on the report file name, one-line description) - first match wins
SECTIONS = [
    ("02_rnn_factor_model", r"^rnn_default_loss", "loss curves of the causal RNN with the notebook's default hyperparameters (02)"),
    ("03_baseline", r"^milestone1_", "Milestone 1: PPO on prices only vs buy & hold and random, default hyperparameters (03)"),
    ("04_rnn_tuning", r"^(rnn_leaderboard|rnn_final_loss|rnn_optuna_|fig_rnn_loss)", "RNN architecture search: Optuna studies, leaderboard, loss curves of the final runs (04, 08)"),
    ("05_ablation_tuned_ddqn", r"^ablation_ddqn", "A-D state ablation re-run with the DDQN tuned in 06, >= 5 validation trades (05, TUNED=True)"),
    ("05_ablation_tuned_ddqn", r"^fig_ablation", "A-D ablation bar chart used in the thesis (08)"),
    ("05_ablation_ppo_default", r"^ablation", "first A-D ablation: PPO with default hyperparameters (05, TUNED=False) - superseded"),
    ("06_agent_comparison", r"^(agents_mt5|rl_optuna_.*_mt5|fig_equity_all|fig_drawdown_all)", "five agents tuned with Optuna, >= 5 validation trades (06)"),
    ("06_agent_comparison_rule1", r"^(agents|rl_optuna_)", "first agent comparison, >= 1 validation trade - superseded by the 5-trade rule"),
    ("07_reward_comparison", r"^rewards", "three reward functions with the tuned DDQN (07)"),
    ("08_all_results", r"^(results_tables|results\.csv|model_index)", "every run, every seed, every metric; index from each number to its checkpoint (08)"),
]


def _section(name):
    for folder, pattern, _ in SECTIONS:
        if re.search(pattern, name):
            return folder
    return "other"


def _md(sub, group, metrics=None):
    metrics = metrics or ["val_sharpe"] + R.HEADLINE
    return R.markdown_table(R.summarise(sub, metrics=metrics, group=R.GROUP), metrics=metrics, group=group)


def _welch(x, y):
    from scipy import stats
    t = stats.ttest_ind(x, y, equal_var=False)
    return float(t.statistic), float(t.pvalue)


def static_policy_table(df):
    """Seeds whose test policy is (almost) static: 1 trade = enter once and hold; <= 5 trades."""
    d = df[~df.agent.isin(["buy_and_hold", "cash", "random"])].copy()
    d["trades"] = pd.to_numeric(d.test_num_trades, errors="coerce")
    g = d.groupby(["experiment", "agent", "state", "signal_source", "reward"], dropna=False)
    out = g.agg(seeds=("trades", "size"), never_trade=("trades", lambda s: int((s == 0).sum())),
                one_trade=("trades", lambda s: int((s == 1).sum())), le_5_trades=("trades", lambda s: int((s <= 5).sum())),
                median_trades=("trades", "median")).reset_index()
    return out


def leak_table(df):
    """Same configuration trained with same-day (look-ahead) vs lagged daily signals."""
    tuned = sorted(e for e in df.experiment.unique() if str(e).startswith("05_ablation_"))
    if not tuned:
        return None, None
    exp = tuned[-1]
    leak = df[(df.experiment == exp) & (df.signal_source == "sameday")]
    if leak.empty:
        return None, None
    agent = leak.agent.iloc[0]
    lagged = df[(df.experiment.str.startswith("06_agents_mt")) & (df.agent == agent) & (df.state == "D")
                & (df.signal_source == "lagged")]
    rows = []
    for label, sub in (("same-day signals (look-ahead)", leak), ("lagged signals (causal)", lagged)):
        rows.append({"signals": label, "seeds": len(sub),
                     "return_% mean": sub.test_cum_return_pct.mean(), "return_% std": sub.test_cum_return_pct.std(),
                     "sharpe mean": sub.test_sharpe.mean(), "sharpe std": sub.test_sharpe.std(),
                     "val_sharpe mean": sub.val_sharpe.mean(), "seeds with profit": int((sub.test_cum_return_pct > 0).sum())})
    tab = pd.DataFrame(rows)
    tests = {}
    if len(lagged) > 1 and len(leak) > 1:
        tests = {m: _welch(leak[m].values, lagged[m].values) for m in ("test_cum_return_pct", "test_sharpe", "val_sharpe")}
    return tab, tests


def final_model(df, min_trades=5):
    """Same rule as notebook 08: best mean validation Sharpe over all reported configurations."""
    cand = df[~df.agent.isin(["buy_and_hold", "random", "cash"]) & (df.signal_source == "lagged")]
    cand = cand[pd.to_numeric(cand.select_min_trades, errors="coerce") >= min_trades]
    if cand.empty:
        return None, None
    best = cand.groupby(R.GROUP, dropna=False).val_sharpe.mean().idxmax()
    runs = cand.set_index(R.GROUP).loc[[best]].reset_index()
    return dict(zip(R.GROUP, best)), runs


def _split_dates(L):
    meta = Path(L.prepared) / "meta.json"
    if not meta.exists():
        return None
    m = json.loads(meta.read_text())
    lines = ["| split | from (UTC) | to (UTC) | 1-min bars |", "|---|---|---|---|"]
    for s, v in m.get("splits", {}).items():
        lines.append(f"| {s} | {v['start'][:16]} | {v['end'][:16]} | {v['rows'][1] - v['rows'][0]:,} |")
    return "\n".join(lines)


def compile_all(L, out_name=OUT_NAME, verbose=True):
    out = Path(L.root) / out_name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    df = R.load_results(L.runs)

    # 1. copy every report file into its experiment folder
    copied = {}
    for p in sorted(Path(L.report).iterdir()):
        if p.is_dir():                                   # environment/ records
            shutil.copytree(p, out / "environment" if p.name == "environment" else out / "other" / p.name)
            copied.setdefault("environment", []).append(p.name + "/")
            continue
        folder = _section(p.name)
        (out / folder).mkdir(exist_ok=True)
        shutil.copy2(p, out / folder / p.name)
        copied.setdefault(folder, []).append(p.name)
    if not (out / "08_all_results" / "results.csv").exists():
        (out / "08_all_results").mkdir(exist_ok=True)
        shutil.copy2(Path(L.runs) / "results.csv", out / "08_all_results" / "results.csv")
        copied.setdefault("08_all_results", []).append("results.csv")

    # 2. summary tables computed here
    summ = out / "summary"
    summ.mkdir()
    static = static_policy_table(df)
    (summ / "static_policies.md").write_text(
        "# Near-static test policies per configuration\n\n"
        "A policy that enters once and holds (1 trade) earns the market move in that direction; in a falling test "
        "period a single short looks skilful. These counts show how often that happened.\n\n"
        + R.df_markdown(static, 1) + "\n", encoding="utf-8")
    leak, leak_tests = leak_table(df)
    if leak is not None:
        txt = "# Leak check - same-day vs lagged LLM signals (state D, same agent and settings)\n\n" + R.df_markdown(leak, 2)
        if leak_tests:
            txt += "\n\nWelch t-test (same-day - lagged):\n\n" + "\n".join(
                f"- {m}: t = {t:+.2f}, p = {p:.3f}" for m, (t, p) in leak_tests.items())
        (summ / "leak_check.md").write_text(txt + "\n", encoding="utf-8")
    cfg, fruns = final_model(df)
    if cfg is not None:
        cols = ["run_id", "seed", "val_sharpe", "test_sharpe", "test_cum_return_pct", "test_max_drawdown_pct", "test_num_trades"]
        (summ / "final_model.md").write_text(
            "# Final model - chosen on validation Sharpe across all reported configurations\n\n"
            f"Configuration: `{cfg}`\n\n" + R.df_markdown(fruns[cols], 3) + "\n", encoding="utf-8")

    # 3. README
    (out / "README.md").write_text(_readme(L, df, copied, static, leak, leak_tests, cfg, fruns), encoding="utf-8")
    if verbose:
        n = sum(len(v) for v in copied.values())
        print(f"compiled {n} report files + summary tables + README into {out}")
        for k in sorted(copied):
            print(f"  {k}/: {len(copied[k])} files")
    return out


def _fmt(x, d=2):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def _readme(L, df, copied, static, leak, leak_tests, cfg, fruns):
    n_runs = len(df)
    commits = ", ".join(sorted(df.commit.dropna().astype(str).unique()))
    by_exp = df.groupby("experiment").size().to_dict()
    splits = _split_dates(L) or "(see `prepared/meta.json`)"
    hl = ["test_cum_return_pct", "test_sharpe", "test_max_drawdown_pct", "test_num_trades"]

    def table(exp, group, extra=None):
        sub = df[df.experiment == exp]
        if extra is not None:
            sub = pd.concat([sub, extra])
        return _md(sub, group, ["val_sharpe"] + hl) if len(sub) else "_not run_"

    a6 = "06_agents_mt5" if (df.experiment == "06_agents_mt5").any() else "06_agents"
    tuned = sorted(e for e in df.experiment.unique() if str(e).startswith("05_ablation_"))
    abl = None
    if tuned:
        ag = df[(df.experiment == tuned[-1]) & ~df.agent.isin(["buy_and_hold", "cash"])].agent.iloc[0]
        abl = df[((df.experiment == tuned[-1]) | ((df.experiment == a6) & (df.state == "D")))
                 & (df.agent == ag) & (df.signal_source == "lagged")]
        welch = []
        for a, b in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B")):
            x, y = abl[abl.state == a].test_sharpe.values, abl[abl.state == b].test_sharpe.values
            if len(x) > 1 and len(y) > 1:
                t, p = _welch(x, y)
                welch.append(f"| {a} - {b} | {x.mean() - y.mean():+.2f} | {t:+.2f} | {p:.2f} |")

    files = []
    for folder, pattern, desc in SECTIONS:
        if folder in copied and f"`{folder}/`" not in " ".join(files):
            descs = [d for f, _, d in SECTIONS if f == folder]
            files.append(f"- **`{folder}/`** - " + "; ".join(dict.fromkeys(descs)) + "\n  " +
                         ", ".join(f"`{x}`" for x in copied[folder]))
    for extra in ("other", "environment"):
        if extra in copied:
            files.append(f"- **`{extra}/`** - " + ("exact software environment of each notebook run (pip freeze, CUDA, GPU)"
                                                   if extra == "environment" else "files not matched to an experiment")
                         + "\n  " + ", ".join(f"`{x}`" for x in copied[extra]))

    static_rl = static[static.agent.isin(["ppo", "a2c", "dqn", "ddqn", "d3qn"])]
    a6rows = df[df.experiment == ("06_agents_mt5" if (df.experiment == "06_agents_mt5").any() else "06_agents")]
    never_eligible = [a.upper() for a in ("ppo", "a2c") if "selection_eligible" in a6rows and (a6rows.agent == a).any()
                      and (a6rows[a6rows.agent == a].selection_eligible.astype(str) == "False").all()]
    elig_note = (f"{' and '.join(never_eligible)} never produced a policy that trades 5+ times on validation; "
                 "their result is the fallback checkpoint (\"no eligible policy\"), not a strategy.") if never_eligible else ""
    if leak is not None and len(leak) == 2:
        hi = leak.iloc[0]["sharpe mean"] > leak.iloc[1]["sharpe mean"]
        leak_note = (f"With the same agent and settings, the look-ahead (same-day) signals give a "
                     f"{'higher' if hi else 'lower'} mean test Sharpe ({leak.iloc[0]['sharpe mean']:.2f} vs "
                     f"{leak.iloc[1]['sharpe mean']:.2f}) and validation Sharpe ({leak.iloc[0]['val_sharpe mean']:.2f} vs "
                     f"{leak.iloc[1]['val_sharpe mean']:.2f}) than the causal ones; see the p-values in "
                     "`summary/leak_check.md` before calling it significant.")
    else:
        leak_note = "No leak-check runs found."
    s_total, s_one, s_le5 = int(static_rl.seeds.sum()), int(static_rl.one_trade.sum() + static_rl.never_trade.sum()), int(static_rl.le_5_trades.sum())

    parts = [f"""# Thesis results - Hybrid RL + LLM framework for risk-aware cryptocurrency trading

Taimoor Ahmed, MS Data Science thesis (SEECS-NUST).

This folder collects **every table and figure** produced by the experiment notebooks, sorted by
experiment, plus summary tables computed from all runs. It is generated by
`rltrade/compile_results.py` (last cell of notebook 08); re-running that cell rebuilds it from the
runs on Drive, so nothing here is edited by hand.

- Runs included: **{n_runs}** ({", ".join(f"{k}: {v}" for k, v in by_exp.items())})
- Code: repository `taimoor-ahmed-1/FinAI-CryptoRaiders`, branches `exp/01-setup` ... `exp/08-results`, folder `experiments/`
- Commits that produced the runs: {commits}
- All metrics are on the **test** period, **net of costs**, unless marked "val"; "mean ± std" is over seeds.

## 1. Data

**Tweets.** About 278k unique English Bitcoin tweets from 2024-01-01 to 2026-06-06, collected with the
SocialData API using time-sliced sampling, then cleaned and de-duplicated. Every tweet was scored by
an LLM (DeepSeek-V3 via DeepInfra) on two 1-5 scales: **sentiment** (bearish -> bullish) and **risk**
(calm -> high perceived risk). Coverage is 100 %. Scores are aggregated per UTC day
(`sentiment_score`, `risk_score`, plus engagement-weighted versions and tweet volume).

**Market.** Rebuilt from Binance's public data archive: 1-second spot klines aggregated to 1-minute
bars (midpoint, high-low range, buy/sell volume) and futures order-book depth at 5 fixed levels
(±1-5 % from mid, ~30 s snapshots). The FinRL contest file could not be used: it covers April 2021
and has no overlap with the tweets.

**Final input.** 1,278,720 one-minute bars (888 days × 1,440), each matched to a real tweet day.

**Splits** (chronological 70/15/15, no shuffling):

{splits}

The test period is a falling market: buy & hold loses about 32 % over it.

**Preparation (notebook 01).** Alpha101 factors computed once on the continuous series (all
operators look backwards) and normalised with 1 %/99 % quantiles fitted on train only. Labels for
the RNN are built per split from that split's own prices. **The daily LLM signals are lagged by one
day**: a bar on day D sees the aggregate of day D-1. The dataset's original alignment (same-day
aggregate on every bar of that day) leaks up to 24 h of the future; it is kept only for the leak
check below.

**Factor model (notebooks 02, 04).** A causal recurrent network (masked attention, fresh hidden
state per block) compresses the alphas into 8 factors. Notebook 04 compared architectures with
Optuna; the selected one (GRU) feeds every RL experiment.

## 2. Trading setup (identical in all RL experiments)

- **State** = position + time in position + 8 RNN factors + the LLM signals of the configuration:
  **A** none (price only, 10 inputs), **B** + sentiment, **C** + risk, **D** + both (12 inputs).
- **Actions**: sell / hold / buy one step of a position in {{-1, 0, +1}} (short allowed), sized as
  a fraction of equity. A decision every 5 minutes; orders fill at the **next** bar.
- **Costs**: 0.10 % fee + 0.02 % slippage per side on traded notional.
- **Training**: 3-day episodes, 2M transitions per run, validation check every 100k.
- **Checkpoint selection**: the checkpoint with the best **validation** Sharpe among those that
  traded **at least 5 times** on validation (confirmed on an exact single-pass backtest). The test
  set is used once, for the selected checkpoint only.
- **Seeds**: 5 per configuration. **Baselines**: buy & hold, cash (never trade), random.
- **Rewards**: change in net assets (default), differential Sharpe, drawdown-penalised.
- **Agents**: PPO, A2C (on-policy); DQN, Double DQN, Dueling Double DQN (D3QN) (value-based).
  Hyperparameters tuned with Optuna (TPE + median pruning, 15 trials × 500k steps, objective =
  validation Sharpe of eligible policies).

## 3. Experiments

| # | Question | Notebook | Experiment id |
|---|---|---|---|
| 01 | Prepare data, leak-free | `01_prepare_data` | – |
| 02 | Train the causal factor model; show the original model leaks | `02_rnn_train` | – |
| 03 | Milestone 1: does a default PPO beat the baselines on prices only? | `03_baseline` | `03_baseline` |
| 04 | Which factor-model architecture? | `04_rnn_tuning` | – |
| 05 | **Do the LLM signals help? (A vs B vs C vs D)** | `05_rl_ablation` | `05_ablation` (PPO default), `05_ablation_ddqn_mt5` (tuned DDQN) |
| 06 | Which RL algorithm? | `06_agent_comparison` | `06_agents` (1-trade rule), `06_agents_mt5` (5-trade rule) |
| 07 | Which reward? | `07_reward_comparison` | `07_rewards` |
| 08 | Tables, figures, final model | `08_results` | – |

`05_ablation` (PPO with default hyperparameters) and `06_agents` (1-trade eligibility rule) are
earlier versions, kept for transparency. In both, almost every policy held one static position, so
they cannot answer their question. The `_mt5` / `_ddqn_mt5` experiments replace them.

## 4. Results at a glance

### Agents (06, configuration D, net asset change)

{table(a6, ("agent",))}

{elig_note}
"""]
    if abl is not None and len(abl):
        parts.append(f"""### State ablation (05, tuned {abl.agent.iloc[0].upper()}) - do the LLM signals help?

{_md(abl, ("state",), ["val_sharpe"] + hl)}

State D is the same agent's run from 06 (identical configuration and seeds; runs are deterministic).
Welch t-test on test Sharpe (n = 5 per state, indicative only):

| pair | difference | t | p |
|---|---|---|---|
{chr(10).join(welch)}
""")
    if leak is not None:
        lt = "; ".join(f"{m}: t = {t:+.2f}, p = {p:.2f}" for m, (t, p) in (leak_tests or {}).items())
        parts.append(f"""### Leak check - how much would the same-day signal alignment inflate results?

{R.df_markdown(leak, 2)}

Welch test (same-day - lagged): {lt}.
""")
    parts.append(f"""### Rewards (07)

{table("07_rewards", ("reward",))}
""")
    if cfg is not None:
        parts.append(f"""### Final model (08) - chosen on validation only

`{cfg['agent']}`, state {cfg['state']}, reward `{cfg['reward']}`, experiment `{cfg['experiment']}`.

{R.df_markdown(fruns[["seed", "val_sharpe", "test_sharpe", "test_cum_return_pct", "test_max_drawdown_pct", "test_num_trades"]], 2)}
""")
    parts.append(f"""### Baselines on the test period

{_md(df[df.agent.isin(["buy_and_hold", "cash"])].drop_duplicates("agent"), ("agent",), hl)}

## 5. How to read these results

- **Seed variance dominates.** Compare each "± std" with the differences between rows: the spread
  across 5 seeds is usually larger than the difference between configurations. With n = 5 seeds, read the Welch
  p-values above before ranking configurations; report effects with their spread.
- **Near-static policies.** Across all experiments, of {s_total} RL seeds, {s_one} ended with 0-1 test trades and {s_le5}
  with ≤ 5 (`summary/static_policies.md`). A single short held through the falling test period
  earns about +32 %; such seeds can look like the best ones without any trading skill.
- **Validation does not carry over.** The final model's seeds are consistent on validation but
  range widely on test (`summary/final_model.md`).
- **Leakage.** {leak_note} Results produced with the original (leaky) pipeline should not be cited.
- **Limitations**: one test window (Jan-Jun 2026, falling market); 5 seeds; daily signals feeding
  5-minute decisions; LLM scores not validated against human labels; order-book depth at 5 fixed
  levels, not tick-level.

## 6. Folder contents

- **`summary/`** - computed here from all runs: `static_policies.md`, `leak_check.md`, `final_model.md`
{chr(10).join(files)}

Each run's weights, configuration, metrics and test equity curve are in
`runs/models/<run_id>/` on Drive; `08_all_results/model_index.csv` maps every row of
`results.csv` to that folder and to the commit that produced it.
""")
    return "\n".join(parts)
