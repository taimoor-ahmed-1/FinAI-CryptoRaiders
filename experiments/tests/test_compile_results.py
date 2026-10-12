import json, sys, tempfile, types
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rltrade import compile_results as C, results as R

tmp = Path(tempfile.mkdtemp()); L = types.SimpleNamespace(root=tmp, runs=tmp/"runs", report=tmp/"report", prepared=tmp/"prepared")
for p in (L.runs, L.report, L.prepared, L.report/"environment"): p.mkdir(parents=True)
(L.report/"environment"/"06_agent_comparison.json").write_text("{}")
names = ["rnn_default_loss.png","milestone1_baseline.md","milestone1_equity.png","rnn_leaderboard.md","rnn_final_loss_curves.png",
 "rnn_optuna_gru_history.png","ablation.md","ablation_bars.png","ablation_ddqn_mt5.md","ablation_ddqn_mt5_bars.png",
 "agents.md","agents_equity.png","agents_mt5.md","agents_mt5_equity.png","rl_optuna_ppo_history.png","rl_optuna_ppo_mt5_history.png",
 "rewards.md","rewards_equity.png","results_tables.md","results.csv","model_index.csv","fig_ablation.png","fig_equity_all_agents.png",
 "fig_drawdown_all_agents.png","fig_rnn_loss.png","mystery.txt"]
for n in names: (L.report/n).write_text("x")
(L.prepared/"meta.json").write_text(json.dumps({"splits":{"train":{"rows":[0,895104],"start":"2024-01-01 00:00:00+00:00","end":"2025-09-13 14:23:00+00:00"}}}))
rng=np.random.default_rng(0); rows=[]
def add(exp, agent, state="D", reward="net_asset_change", src="lagged", n=5, smt=5, elig="True"):
    for s in range(n):
        r=dict(run_id=f"{exp}/{agent}_{state}_{src}_{reward}_s{s}", experiment=exp, agent=agent, state=state, signal_source=src,
               reward=reward, tag="", seed=s, val_sharpe=rng.normal(1,1), select_min_trades=smt if agent not in ("buy_and_hold","cash") else "",
               commit="abc1234", selection_eligible=("False" if agent in ("ppo","a2c") and exp=="06_agents_mt5" else elig), **{m: rng.normal() for m in R.HEADLINE})
        r["test_num_trades"]=int(rng.integers(0,50)); r["test_cum_return_pct"]=rng.normal(0,30); rows.append(r)
for a in ("ppo","a2c","dqn","ddqn","d3qn"): add("06_agents_mt5", a)
add("06_agents_mt5","buy_and_hold",n=1); add("06_agents_mt5","cash",n=1); add("06_agents","ppo",smt=1)
for st in "ABC": add("05_ablation_ddqn_mt5","ddqn",state=st)
add("05_ablation_ddqn_mt5","ddqn",src="sameday"); add("05_ablation","ppo",state="A",smt="")
for rw in ("net_asset_change","diff_sharpe","drawdown_penalized"): add("07_rewards","ddqn",reward=rw)
pd.DataFrame(rows).to_csv(L.runs/"results.csv", index=False)
out = C.compile_all(L)
readme = (out / "README.md").read_text(encoding="utf-8")


def test_files_sorted_by_experiment():
    assert (out / "05_ablation_tuned_ddqn" / "fig_ablation.png").exists()
    assert (out / "06_agent_comparison" / "rl_optuna_ppo_mt5_history.png").exists()
    assert (out / "06_agent_comparison_rule1" / "rl_optuna_ppo_history.png").exists()
    assert (out / "other" / "mystery.txt").exists() and (out / "environment").is_dir()


def test_summary_tables_and_readme():
    for f in ("static_policies.md", "leak_check.md", "final_model.md"):
        assert (out / "summary" / f).exists()
    assert "## 1. Data" in readme and "Leak check" in readme and "never produced" in readme
    assert "2024-01-01 00:00" in readme                     # split dates from prepared/meta.json
