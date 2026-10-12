"""compile_results builds the numbered figures, tables and README from synthetic runs (no real data needed)."""
import sys
import tempfile
import types
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import compile_results as C  # noqa: E402
from rltrade import results as R  # noqa: E402

tmp = Path(tempfile.mkdtemp())
L = types.SimpleNamespace(root=tmp, runs=tmp / "runs", report=tmp / "report", prepared=tmp / "prepared", rnn_tuning=tmp / "rnn")
for p in (L.runs, L.report, L.prepared, L.rnn_tuning):
    p.mkdir(parents=True)
rng = np.random.default_rng(0)
n = 2000
t = (pd.Timestamp("2026-01-24", tz="UTC").value + np.arange(n) * 60 * 10**9).astype(np.int64)
for split, start in (("train", "2024-01-01"), ("val", "2025-09-13"), ("test", "2026-01-24")):
    tt = (pd.Timestamp(start, tz="UTC").value + np.arange(3 * 1440) * 60 * 10**9).astype(np.int64)
    np.savez(L.prepared / f"market_{split}.npz", midpoint=60000 + np.cumsum(rng.normal(0, 30, len(tt))), time_ns=tt,
             sameday__sentiment_score=rng.uniform(3, 4, len(tt)), sameday__risk_score=rng.uniform(1.5, 2, len(tt)),
             sameday__tweet_volume=rng.integers(250, 350, len(tt)).astype(float))
pd.DataFrame([{"arch": a, "seed": s, "val_mse": 0.165 + rng.normal(0, 1e-4), "val_ic": 0.15, "test_ic": 0.12}
              for a in ("gru", "lstm") for s in range(3)]).to_csv(L.rnn_tuning / "final_runs.csv", index=False)
rows = []


def add(exp, agent, state="D", reward="net_asset_change", src="lagged", k=5):
    for s in range(k):
        rid = f"{exp}/{agent}_{state}_{src}_{reward}_s{s}"
        rows.append(dict(run_id=rid, experiment=exp, agent=agent, state=state, signal_source=src, reward=reward, tag="", seed=s,
                         val_sharpe=rng.normal(1, 1), select_min_trades=5, selection_eligible=agent not in ("ppo", "a2c"),
                         commit="abc1234", **{m: rng.normal() for m in R.HEADLINE}))
        rows[-1]["test_num_trades"] = int(rng.integers(0, 100))
        rows[-1]["test_cum_return_pct"] = rng.normal(0, 25)
        d = L.runs / "models" / rid
        d.mkdir(parents=True)
        np.savez(d / "test_backtest.npz", equity=1 + np.cumsum(rng.normal(0, 1e-3, n)), time_ns=t)


for a in C.RL_AGENTS:
    add(C.EXP_AGENTS, a)
add(C.EXP_AGENTS, "buy_and_hold", k=1)
add(C.EXP_AGENTS, "cash", k=1)
for st in "ABC":
    add(C.EXP_ABLATION, "ddqn", state=st)
add(C.EXP_ABLATION, "ddqn", src="sameday")          # leak-check runs: must be excluded
add("06_agents", "ppo")                              # superseded: must be excluded
for rw in C.REWARD_NAMES:
    add(C.EXP_REWARDS, "ddqn", reward=rw)
pd.DataFrame(rows).to_csv(L.runs / "results.csv", index=False)
out = C.compile_all(L, verbose=False)
readme = (out / "README.md").read_text(encoding="utf-8")


def test_all_numbered_figures_exist():
    names = sorted(p.name for p in out.glob("*.png"))
    assert [n[:2] for n in names] == [f"{i:02d}" for i in range(1, 12)], names


def test_tables_and_readme():
    assert (out / "tables" / "06_ablation.csv").exists() and (out / "tables" / "04_agents.csv").exists()
    for i in range(1, 12):
        assert f"## {i:02d} " in readme
    assert "never learned to trade actively" in readme


def test_leaky_and_superseded_runs_excluded():
    df = C.current_runs(pd.read_csv(L.runs / "results.csv"))
    assert (df.signal_source == "lagged").all() and "06_agents" not in set(df.experiment)
