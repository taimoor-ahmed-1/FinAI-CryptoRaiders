import os
import re
import json
import torch
import numpy as np
import erl_agent
from erl_config import Config, build_env
from data_config import ConfigData
from trade_simulator import EvalTradeSimulator, window_starts
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown, cumulative_returns


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


def member_dirs(models_dir):
    """[fix] every trained member (task1_ensemble saves <idx>_<AgentClass>/); the kit loaded one folder per class
    name, i.e. 3 of the 4 members it had trained"""
    names = sorted(d for d in os.listdir(models_dir) if re.match(r"\d+_Agent\w+$", d))
    return [(os.path.join(models_dir, d), getattr(erl_agent, d.split("_", 1)[1])) for d in names]


class EnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config):
        self.save_path = save_path
        self.agent_classes = agent_classes

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = args.state_dim  # 2 + 8 [+ 2 news signals]
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)

        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        # self.net_assets = [torch.tensor(args.starting_cash, device=self.device)]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash
        self.sim_assets = [args.starting_cash]  # [fix] the simulator's own account (shorts, slippage, stop-loss)
        self.prices = []
        self.trades = 0

    def load_agents(self, models_dir=None):
        args = self.args
        for cwd, agent_class in member_dirs(models_dir or self.save_path):
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent.save_or_load_agent(cwd, if_save=False)  # Load agent
            self.agents.append(agent)

    def multi_trade(self):
        """Evaluation loop using ensemble of agents"""

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]

        for _ in range(trade_env.max_step):
            actions = []
            intermediate_state = last_state

            # Collect actions from each agent
            for agent in agents:
                actor = agent.act
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                tensor_q_values = actor(tensor_state)
                tensor_action = tensor_q_values.argmax(dim=1)
                action = tensor_action.detach().cpu().unsqueeze(1)
                actions.append(action)

            action = self._ensemble_action(actions=actions)
            action_int = action.item() - 1

            # [fix] the row this step trades at, read before step(): the env resets itself inside the last step
            price_row = (trade_env.step_is + trade_env.step_i + trade_env.step_gap)[0].item()
            state, reward, done, _ = trade_env.step(action=action)
            self.sim_assets.append(self.sim_assets[-1] + float(reward[0]))
            self.trades += int(trade_env.action_int[0].abs().item())  # executed, after limits and stop-loss

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # Manually compute cumulative returns
            # [fix] the kit read price_ary[trade_env.step_i] - the episode counter, not the row - so its
            # bookkeeping priced every window with the first 79 minutes of the file
            mid_price = trade_env.price_ary[price_row, 2].to(self.device)
            self.prices.append(float(mid_price))

            new_cash = self.cash[-1]

            if action_int > 0 and self.cash[-1] > mid_price:  # Buy
                last_cash = self.cash[-1]
                new_cash = last_cash - mid_price
                self.current_btc += 1
            elif action_int < 0 and self.current_btc > 0:  # Sell
                last_cash = self.cash[-1]
                new_cash = last_cash + mid_price
                self.current_btc -= 1

            self.cash.append(new_cash)
            self.btc_assets.append((self.current_btc * mid_price).item())
            self.net_assets.append((to_python_number(self.btc_assets[-1]) + to_python_number(new_cash)))

            last_state = state

            # Log win rate
            if action_int == 1:
                correct_pred.append(1 if last_price < mid_price else -1 if last_price > mid_price else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < mid_price else 1 if last_price > mid_price else 0)
            else:
                correct_pred.append(0)

            last_price = mid_price
            current_btcs.append(self.current_btc)

        # Save results
        positions_np = np.array([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in positions])
        np.save(f"{self.save_path}_positions.npy", positions_np)
        #np.save(f"{self.save_path}_positions.npy", positions)
        np.save(f"{self.save_path}_net_assets.npy", np.array(self.net_assets))
        np.save(f"{self.save_path}_btc_positions.npy", np.array(self.btc_assets))
        np.save(f"{self.save_path}_correct_predictions.npy", np.array(correct_pred))
        np.save(f"{self.save_path}_sim_assets.npy", np.array(self.sim_assets))

        # Compute metrics
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        final_sharpe_ratio = sharpe_ratio(returns)
        final_max_drawdown = max_drawdown(returns)
        final_roma = return_over_max_drawdown(returns)

        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")

    def _ensemble_action(self, actions):
        """Returns the majority action among agents. Our code uses majority voting, you may change this to increase performance."""
        count = Counter([a.item() for a in actions])
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)


def curve_metrics(net):
    """The kit's metrics (metrics.py) on an equity curve. Sharpe is per 2-second step, not annualised; it is
    reported as None when the curve is flat (the kit returns inf for 0 / 0)."""
    net = np.asarray(net, dtype=float)
    r = np.diff(net) / net[:-1]
    return {"cum_return_pct": float(cumulative_returns(r)[-1] * 100),
            "sharpe_per_step": None if r.std() == 0 else float(sharpe_ratio(r)),
            "max_drawdown_pct": float(max_drawdown(r) * 100), "steps": int(len(r))}


def run_evaluation(save_path, agent_list=None, gpu_id=None, data_dir=None, split="test", out_dir=None,
                   slippage=7e-7, stop_loss_rate=1e-3, signals=False):
    """[fix] Scores the ensemble on EVERY consecutive 79-minute episode of `split` (the kit scored one random
    episode, of the training file). Curves per episode, each from $1M:
        sim      the simulator's account: position -1/0/+1 BTC, slippage, stop-loss - the headline
        kit      the kit's own bookkeeping (long only, 1 BTC per signal, no costs), price index fixed
        bh_1btc  buy 1 BTC at the episode start and hold (the same dollar exposure as the agent)
        bh_full  all $1M in BTC (context)
    The whole split is scored by chaining the episodes' step returns."""
    import sys
    import pandas as pd

    if gpu_id is None:
        gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # Get GPU_ID from command line arguments
    out_dir = out_dir or f"{save_path}_eval_{split}"
    os.makedirs(f"{out_dir}/episodes", exist_ok=True)

    num_sims = 1
    num_ignore_step = 60
    max_position = 1
    step_gap = 2

    max_step = (4800 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 2 + 8 + (2 if signals else 0),  # [baseline] no LLM signals; [news] + sentiment, risk
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "data_dir": data_dir,  # [fix] the kit's "dataset_path" was never read by the simulator
        "split": split,
        "stop_loss_rate": stop_loss_rate,
        "signals": signals,
    }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6

    cash = args.starting_cash
    r0, r1 = ConfigData(data_dir).split_rows(split)
    starts = window_starts(r1 - r0, max_step * step_gap)
    agents, rows, curves = None, [], {"sim": [], "kit": [], "bh_1btc": [], "bh_full": []}
    for k, start in enumerate(starts):
        ensemble_evaluator = EnsembleEvaluator(f"{out_dir}/episodes/e{k:02d}", agent_list, args)
        ensemble_evaluator.trade_env.fixed_starts = [start]
        if agents is None:
            ensemble_evaluator.load_agents(save_path)
            agents = ensemble_evaluator.agents
        ensemble_evaluator.agents = agents
        ensemble_evaluator.multi_trade()

        p0 = float(ensemble_evaluator.trade_env.price_ary[start, 2])
        prices = np.array(ensemble_evaluator.prices)
        episode = {"sim": ensemble_evaluator.sim_assets, "kit": ensemble_evaluator.net_assets,
                   "bh_1btc": np.r_[cash, cash + prices - p0], "bh_full": np.r_[cash, cash * prices / p0]}
        for name, net in episode.items():
            curves[name].append(np.asarray(net, dtype=float))
            rows.append(dict(curve_metrics(net), curve=name, episode=k, start_row=r0 + start,
                             trades=ensemble_evaluator.trades if name in ("sim", "kit") else 1))
        print(f"| episode {k:2d} row {r0 + start:7d}: sim {rows[-4]['cum_return_pct']:+.4f}%  "
              f"b&h(1 BTC) {rows[-2]['cum_return_pct']:+.4f}%  trades {ensemble_evaluator.trades}")

    per = pd.DataFrame(rows)
    per.to_csv(f"{out_dir}/per_episode.csv", index=False)
    summary = []
    for name, cs in curves.items():
        r = np.concatenate([np.diff(c) / c[:-1] for c in cs])
        net = cash * np.concatenate([[1.0], np.cumprod(1 + r)])
        np.save(f"{out_dir}/equity_{name}.npy", net)
        sub = per[per.curve == name]
        summary.append(dict(curve_metrics(net), curve=name, episodes=len(cs), trades=int(sub.trades.sum()),
                            episode_return_pct_mean=float(sub.cum_return_pct.mean()),
                            episode_return_pct_median=float(sub.cum_return_pct.median()),
                            episodes_positive=int((sub.cum_return_pct > 0).sum())))
    pd.DataFrame(summary).to_csv(f"{out_dir}/summary.csv", index=False)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump({"split": split, "rows": [r0, r1], "members": [d for d, _ in member_dirs(save_path)],
                   "slippage": slippage, "stop_loss_rate": stop_loss_rate, "signals": signals, "curves": summary},
                  f, indent=1)
    print(pd.DataFrame(summary).to_string(index=False))
    return summary


if __name__ == "__main__":
    save_path = "ensemble_teamname/ensemble_models"  # [fix] where task1_ensemble.py saves (was "trained_agents")
    run_evaluation(save_path)
