"""
Runs one experiment configuration for one seed, end to end:

    train on the TRAIN split
      -> every `eval_every` transitions, backtest the current policy on VALIDATION
         and keep the checkpoint with the best validation score (Sharpe by default)
      -> load that checkpoint and backtest it ONCE on TEST (exact, sequential pass)
      -> save weights + config + seed + git commit + metrics, append a results.csv row

The test split never influences training, checkpoint choice or hyperparameters.

Checkpoints that trade fewer than `min_val_trades` times on validation (default 5) are
NOT eligible. With transaction costs and a falling validation market, "stay in cash"
(0 trades, Sharpe 0) and "short once and hold" (1 trade) beat every policy that
actually trades, so selection would reward a do-nothing or one-off directional bet
rather than trading. Staying in cash is reported as its own baseline ("cash").

Every run has a stable id, e.g. `05_ablation/ppo_D_net_asset_change_s3`, which is
both its folder under `<out>/models/` and its `run_id` in results.csv, so any number
in the report traces back to one checkpoint. Runs whose metrics.json already exists
are skipped, so a notebook can simply be re-run after a Colab disconnect.
"""
import csv
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch as th

from . import seeding
from .agents import make_agent
from .baselines import BASELINES, make_baseline
from .env import EnvConfig, Market, TradingEnv, run_policy
from .metrics import compute_metrics

INVALID_SCORE = -1e6        # score of a checkpoint that is not eligible (trades too little)
SELECTION_RULE = 2          # bump when the eligibility rule changes (part of Optuna study names)


def selection_score(metrics, metric="sharpe", min_trades=1):
    """(score, eligible) of a validation backtest. Never-trading policies are ineligible."""
    eligible = metrics["num_trades"] >= min_trades
    return (metrics[metric] if eligible else INVALID_SCORE), eligible


class CheckpointSelector:
    """Keeps the best ELIGIBLE checkpoint seen during training.

    Validation during training is screened with a fast chunked backtest (the split
    cut into segments run in parallel). Every segment starts flat, so a policy that
    just enters once and holds shows one trade PER SEGMENT there and would pass a
    trade-count rule it does not meet. So a checkpoint that could become the new best
    is confirmed on the exact single-pass backtest, and both its score and its trade
    count come from that pass. Checkpoints that cannot be a new best are not confirmed.
    """

    def __init__(self, metric="sharpe", min_trades=5):
        self.metric, self.min_trades = metric, min_trades
        self.score, self.step, self.state, self.eligible = -np.inf, 0, None, None

    def consider(self, chunked, exact, get_state, steps):
        """chunked: metrics of the chunked backtest; exact: () -> metrics of the single pass.

        Returns (eligible, exact_metrics_or_None); eligible is None when not confirmed.
        """
        exact_m, eligible, score = None, None, None
        if chunked["num_trades"] < self.min_trades:          # cannot meet the rule in a single pass either
            eligible, score = False, INVALID_SCORE
        elif chunked[self.metric] > self.score or self.state is None:
            exact_m = exact()
            score, eligible = selection_score(exact_m, self.metric, self.min_trades)
        if score is not None and (score > self.score or self.state is None):
            self.score, self.step, self.eligible, self.state = score, steps, eligible, get_state()
        return eligible, exact_m


METRIC_KEYS = ("cum_return_pct", "annual_return_pct", "sharpe", "sortino", "max_drawdown_pct", "romad",
               "win_rate_pct", "round_trips", "num_trades", "turnover", "exposure_pct", "long_pct",
               "short_pct", "costs_pct", "steps")


@dataclass
class RunSpec:
    experiment: str                      # e.g. "03_baseline"
    agent: str                           # ppo / a2c / dqn / ddqn / d3qn / buy_and_hold / random
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    agent_hp: dict = field(default_factory=dict)
    total_steps: int = 2_000_000         # transitions (notebook: TOTAL_STEPS = 2e6)
    eval_every: int = 100_000            # transitions between validation checks
    val_chunks: int = 16                 # parallel segments for the in-training validation backtest
    select_metric: str = "sharpe"
    min_val_trades: int = 5              # checkpoints that trade less on validation are ineligible
    tag: str = ""                        # optional extra label in the run id

    @property
    def run_id(self):
        signals = self.env.state if self.env.signals is None else "+".join(self.env.signals)
        if self.env.signal_source != "lagged":
            signals += f"-{self.env.signal_source}"
        parts = [self.agent, signals, self.env.reward] + ([self.tag] if self.tag else []) + [f"s{self.seed}"]
        return f"{self.experiment}/" + "_".join(parts)


_MARKETS = {}


def _market(prepared_dir, factor_dir, split, cfg, device):
    key = (str(prepared_dir), str(factor_dir), split, cfg.signal_cols, cfg.signal_source, str(device))
    if key not in _MARKETS:
        _MARKETS[key] = Market(prepared_dir, factor_dir, split, cfg, device)
    return _MARKETS[key]


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def append_result(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    if not new:                                           # keep one row per run id
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run_id"] != row["run_id"]]
            fields = list(rows[0].keys()) if rows else list(row.keys())
        for k in row:
            if k not in fields:
                fields.append(k)
        rows.append(row)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    else:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)


def run(spec, prepared_dir, factor_dir, out_root, device="auto", overwrite=False,
        trial=None, save=True, verbose=True):
    """Train (if needed), select on validation, test once. Returns the results row.

    trial: optional Optuna trial - the validation score is reported at every check
           and the run is pruned when Optuna says so (used for hyperparameter search).
    save:  False skips writing model files and results.csv (tuning trials).
    """
    out_root = Path(out_root)
    out = out_root / "models" / spec.run_id
    if save and not overwrite and (out / "metrics.json").exists():
        row = json.loads((out / "metrics.json").read_text())["row"]
        if verbose:
            print(f"[skip] {spec.run_id} (already done)")
        return row

    t0 = time.time()
    dev = seeding.pick_device(device)
    seeding.set_seed(spec.seed)
    cfg = spec.env
    mk = {s: _market(prepared_dir, factor_dir, s, cfg, dev) for s in ("train", "val", "test")}
    ppy = cfg.periods_per_year
    val_curve = []
    sel = CheckpointSelector(spec.select_metric, spec.min_val_trades)

    if spec.agent in BASELINES:
        policy = make_baseline(spec.agent, spec.seed)
        agent_cfg = {}
    else:
        agent = make_agent(spec.agent, cfg.state_dim, 3, dev, spec.seed, **spec.agent_hp)
        agent_cfg = agent.config()
        env = TradingEnv(mk["train"], cfg, num_envs=agent.cfg.num_envs, mode="train", seed=spec.seed)

        def callback(ag, steps):
            pol = lambda o: ag.act(o, deterministic=True)       # noqa: E731
            m = compute_metrics(run_policy(pol, mk["val"], cfg, chunks=spec.val_chunks), ppy)
            snapshot = lambda: {k: {n: t.detach().cpu().clone() for n, t in v.items()}   # noqa: E731
                                for k, v in ag.state_dict().items()}
            eligible, exact_m = sel.consider(m, lambda: compute_metrics(run_policy(pol, mk["val"], cfg, chunks=1), ppy),
                                             snapshot, steps)
            val_curve.append({"steps": steps, "eligible": eligible,
                              **{k: m[k] for k in ("sharpe", "cum_return_pct", "max_drawdown_pct", "num_trades")},
                              **({"exact_sharpe": exact_m["sharpe"], "exact_trades": exact_m["num_trades"]}
                                 if exact_m else {})})
            if verbose:
                if exact_m is not None:
                    note = (f" | exact pass: sharpe {exact_m['sharpe']:+.2f} trades {exact_m['num_trades']}"
                            + ("" if eligible else f" (< {spec.min_val_trades} trades - not eligible)"))
                elif eligible is False:
                    note = f" (< {spec.min_val_trades} trades - not eligible)"
                else:
                    note = ""
                best_txt = f"{sel.score:+.2f}" if sel.eligible else "none eligible yet"
                print(f"  {spec.run_id} | {steps:>9,} steps | val sharpe {m['sharpe']:+.2f} "
                      f"ret {m['cum_return_pct']:+.1f}% trades {m['num_trades']}{note} | best {best_txt}")
            if trial is not None:
                trial.report(sel.score, steps)                 # running best eligible score
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            return False

        agent.learn(env, spec.total_steps, callback=callback, callback_every=spec.eval_every)
        if not val_curve or val_curve[-1]["steps"] < spec.total_steps - spec.eval_every // 2:
            callback(agent, spec.total_steps)                  # always score the final policy too
        agent.load_state_dict(sel.state)
        policy = lambda o: agent.act(o, deterministic=True)   # noqa: E731

    bts = {s: run_policy(policy, mk[s], cfg, chunks=1) for s in ("val", "test")}
    mets = {s: compute_metrics(bts[s], ppy) for s in bts}
    info = seeding.run_info(spec.seed)
    row = {"run_id": spec.run_id, "experiment": spec.experiment, "agent": spec.agent,
           "state": cfg.state if cfg.signals is None else "+".join(cfg.signals),
           "signal_source": cfg.signal_source, "reward": cfg.reward, "tag": spec.tag, "seed": spec.seed,
           "state_dim": cfg.state_dim, "total_steps": spec.total_steps if spec.agent not in BASELINES else 0,
           "best_step": sel.step if spec.agent not in BASELINES else 0, "select_metric": spec.select_metric,
           "select_min_trades": spec.min_val_trades if spec.agent not in BASELINES else "",
           "selection_eligible": sel.eligible if spec.agent not in BASELINES else "",
           "selection_rule": SELECTION_RULE,
           **{f"val_{k}": mets["val"][k] for k in METRIC_KEYS},
           **{f"test_{k}": mets["test"][k] for k in METRIC_KEYS},
           "fee_rate": cfg.fee_rate, "slippage": cfg.slippage, "step_gap": cfg.step_gap,
           "minutes": round((time.time() - t0) / 60, 2), "commit": info["commit"], "device": info["device"],
           "agent_hp": json.dumps(_jsonable(agent_cfg), sort_keys=True)}
    if verbose:
        print(f"[done] {spec.run_id} | test sharpe {mets['test']['sharpe']:+.2f} "
              f"ret {mets['test']['cum_return_pct']:+.2f}% mdd {mets['test']['max_drawdown_pct']:.1f}% "
              f"trades {mets['test']['num_trades']} | {row['minutes']} min")
    if save:
        out.mkdir(parents=True, exist_ok=True)
        if spec.agent not in BASELINES:
            th.save(sel.state, out / "model.pt")
            (out / "history.json").write_text(json.dumps(_jsonable({"train": agent.history,
                                                                      "val_curve": val_curve})))
        spec_d = dataclasses.asdict(spec)
        (out / "config.json").write_text(json.dumps(_jsonable({"spec": spec_d, "agent_config": agent_cfg,
                                                               "prepared_dir": str(prepared_dir),
                                                               "factor_dir": str(factor_dir)}), indent=2))
        np.savez_compressed(out / "test_backtest.npz", **{k: bts["test"][k] for k in
                                                          ("equity", "level", "time_ns", "returns")})
        (out / "metrics.json").write_text(json.dumps(_jsonable({"row": row, "val": mets["val"],
                                                                "test": mets["test"], "run": info}), indent=2))
        append_result(out_root / "results.csv", row)
    return row


def run_seeds(spec, seeds, prepared_dir, factor_dir, out_root, **kw):
    """The same configuration over several seeds (brief: at least 5)."""
    return [run(dataclasses.replace(spec, seed=s), prepared_dir, factor_dir, out_root, **kw) for s in seeds]
