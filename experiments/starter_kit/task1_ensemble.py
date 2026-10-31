import os
import json
import time
from copy import deepcopy
import torch
import numpy as np
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_evaluator import Evaluator
from trade_simulator import TradeSimulator, EvalTradeSimulator, window_starts
from data_config import ConfigData
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter

from metrics import *


def can_buy(action, mid_price, cash, current_btc):
    if action == 1 and cash > mid_price:  # can buy
        last_cash = cash
        new_cash = last_cash - mid_price
        current_btc += 1
    elif action == -1 and current_btc > 0:  # can sell
        last_cash = cash
        new_cash = last_cash + mid_price
        current_btc -= 1
    else:
        new_cash = cash

    return new_cash, current_btc


def winloss(action, last_price, mid_price):
    if action > 0:
        if last_price < mid_price:
            correct_pred = 1
        elif last_price > mid_price:
            correct_pred = -1
        else:
            correct_pred = 0
    elif action < 0:
        if last_price < mid_price:
            correct_pred = -1
        elif last_price > mid_price:
            correct_pred = 1
        else:
            correct_pred = 0
    else:
        correct_pred = 0
    return correct_pred


class Ensemble:
    def __init__(self, log_rules, save_path, starting_cash, agent_classes, args: Config):

        self.log_rules = log_rules

        # ensemble configs
        self.save_path = save_path
        self.starting_cash = starting_cash
        self.current_btc = 0
        self.position = [0]
        self.btc_assets = [0]
        self.net_assets = [starting_cash]
        self.cash = [starting_cash]
        self.agent_classes = agent_classes

        self.from_env_step_is = None

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = args.state_dim  # 2 + 8 [+ 2 news signals]
        # gpu_id = 0
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        eval_env_class = args.eval_env_class
        eval_env_class.num_envs = 1

        eval_env_args = args.eval_env_args
        eval_env_args["num_envs"] = 1
        eval_env_args["num_sims"] = 1

        self.trade_env = build_env(eval_env_class, eval_env_args, gpu_id=args.gpu_id)

        self.actions = []

        self.firstbpi = True

    def member_dir(self, idx):
        # [fix] one folder per member: the kit used the class name, so the second AgentDoubleDQN overwrote
        # the first, and task1_eval.py then loaded only 3 of the 4 trained agents
        return os.path.join(self.save_path, "ensemble_models", f"{idx}_{self.agent_classes[idx].__name__}")

    def save_member(self, idx, agent):
        agent_dir = self.member_dir(idx)
        os.makedirs(agent_dir, exist_ok=True)
        agent.save_or_load_agent(agent_dir, if_save=True)
        with open(os.path.join(agent_dir, "selection.json"), "w") as f:  # written last: marks the member done
            json.dump(agent.selection, f, indent=1)

    def save_ensemble(self):
        """Saves the ensemble of agents to a directory."""
        for idx, agent in enumerate(self.agents):
            self.save_member(idx, agent)
        print(f"Ensemble models saved in directory: {os.path.join(self.save_path, 'ensemble_models')}")

    def ensemble_train(self):
        args = self.args

        base_seed = args.random_seed
        for idx, agent_class in enumerate(self.agent_classes):
            if os.path.exists(os.path.join(self.member_dir(idx), "selection.json")):
                # [fix] each member is saved as soon as it is trained, so a re-run after a Colab disconnect
                # resumes with the next member (the kit saved all of them only at the very end)
                print(f"| member {idx} {agent_class.__name__} already trained, skipped")
                continue

            args.agent_class = agent_class
            # [fix] a seed and a folder per member. The kit gave every member the same seed and, because
            # Config.cwd is set only once, the same training folder; AgentDoubleDQN (twice) and AgentTwinD3QN
            # (the same network, QNetTwin) therefore trained into identical weights.
            args.random_seed = base_seed + idx
            args.cwd = os.path.join(self.save_path, f"train_{idx}_{agent_class.__name__}")

            agent = self.train_agent(args=args)
            self.save_member(idx, agent)

        print(f"Ensemble models saved in directory: {os.path.join(self.save_path, 'ensemble_models')}")

    def _majority_vote(self, actions):
        """handles tie breaks by returning first element of the most common ones"""
        count = Counter(actions)
        majority_action, _ = count.most_common(1)[0]
        return majority_action

    def train_agent(self, args: Config):
        """
        Trains agent
        Builds env inside
        """
        args.init_before_training()
        torch.set_grad_enabled(False)

        """init environment"""
        env = build_env(args.env_class, args.env_args, args.gpu_id)

        """init agent"""
        agent = args.agent_class(
            args.net_dims,
            args.state_dim,
            args.action_dim,
            gpu_id=args.gpu_id,
            args=args,
        )
        agent.save_or_load_agent(args.cwd, if_save=False)

        state = env.reset()

        if args.num_envs == 1:
            assert state.shape == (args.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            if state.shape != (args.num_envs, args.state_dim):
                raise ValueError(f"state.shape == (num_envs, state_dim): {state.shape, args.num_envs, args.state_dim}")
            if not isinstance(state, torch.Tensor):
                raise TypeError(f"isinstance(state, torch.Tensor): {repr(state)}")
            state = state.to(agent.device)
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        agent.last_state = state.detach()

        """init buffer"""

        if args.if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.gpu_id,
                num_seqs=args.num_envs,
                max_size=args.buffer_size,
                state_dim=args.state_dim,
                action_dim=1 if args.if_discrete else args.action_dim,
            )
            buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
            buffer.update(buffer_items)  # warm up for ReplayBuffer
        else:
            buffer = []

        """init evaluator"""
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = dict(args.eval_env_args if args.eval_env_args else args.env_args)
        # [fix] score every consecutive episode of the validation split, the same windows each time. The kit's
        # Ensemble.__init__ had set eval_env_args to 1 env in place, so it scored 3 random windows per round.
        r0, r1 = ConfigData(eval_env_args.get("data_dir")).split_rows(eval_env_args["split"])
        n_val = r1 - r0
        val_starts = window_starts(n_val, args.max_step * eval_env_args["step_gap"])
        eval_env_args.update(num_envs=len(val_starts), num_sims=len(val_starts))
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        eval_env.fixed_starts = val_starts
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args)
        evaluator.eval_times = len(val_starts)  # one pass over the windows (not args: it sizes the warm-up)

        """train loop"""
        cwd = args.cwd
        break_step = args.break_step
        horizon_len = args.horizon_len
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        args_seed = args.random_seed
        del args

        import torch as th

        # [fix] keep the actor with the best VALIDATION return. The kit's evaluator scored on the training data
        # and its best checkpoint was never used: the final weights were saved whatever happened.
        best_val, best_state, best_step = -float("inf"), None, 0

        if_train = True
        while if_train:
            buffer_items = agent.explore_env(env, horizon_len)

            action = buffer_items[1].flatten()
            action_count = th.bincount(action).data.cpu().numpy() / action.shape[0]
            action_count = np.ceil(action_count * 998).astype(int)

            position = buffer_items[0][:, :, 0].long().flatten()
            position = position.float()  # TODO Only if on cpu
            position_count = torch.histc(position, bins=env.max_position * 2 + 1, min=-2, max=2)
            position_count = position_count.data.cpu().numpy() / position.shape[0]
            position_count = np.ceil(position_count * 998).astype(int)

            print(";;;", " " * 70, action_count, position_count)

            exp_r = buffer_items[2].mean().item()
            if if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            evaluator.evaluate_and_save(
                actor=agent.act,
                steps=horizon_len,
                exp_r=exp_r,
                logging_tuple=logging_tuple,
            )
            val_r = evaluator.recorder[-1][1]  # mean episode return on the validation split
            if val_r > best_val:
                best_val, best_step = val_r, evaluator.total_step
                best_state = deepcopy(agent.act.state_dict())
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

        print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}")
        agent.act.load_state_dict(best_state)  # act is also the critic (cri_class is None)
        agent.selection = {"best_step": int(best_step), "best_val_return": float(best_val),
                           "total_step": int(evaluator.total_step), "seed": int(args_seed),
                           "val_curve": [[int(r[0]), float(r[1])] for r in evaluator.recorder]}
        print(f"| best validation return {best_val:.2f} at step {best_step:.2e}")

        env.close() if hasattr(env, "close") else None
        evaluator.save_training_curve_jpg()
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, "save_or_load_history"):
            buffer.save_or_load_history(cwd, if_save=True)

        self.from_env_step_is = env.step_is
        return agent


def run(save_path, agent_list, log_rules=False, gpu_id=None, break_step=int(32e4), num_sims=2**12,
        slippage=7e-7, seed=0, data_dir=None, stop_loss_rate=1e-3, signals=False):
    """[fix] the settings are arguments; defaults are the kit's, with break_step restored to 32e4 (its TODO)"""
    import sys

    if gpu_id is None:
        gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # Get GPU_ID from command line parameters

    from erl_agent import AgentD3QN

    num_ignore_step = 60
    max_position = 1
    step_gap = 2

    max_step = (4800 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 2 + 8 + (2 if signals else 0),  # (position, holding) + factor_dim [+ news; baseline: none]
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "data_dir": data_dir,
        "split": "train",
        "stop_loss_rate": stop_loss_rate,
        "signals": signals,  # [news]
    }
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = seed  # [fix] was gpu_id (the same for every member); +member index in ensemble_train
    args.net_dims = (128, 128, 128)

    args.gamma = 0.995
    args.explore_rate = 0.005
    args.state_value_tau = 0.01
    args.soft_update_tau = 2e-6
    args.learning_rate = 2e-6
    args.batch_size = 512
    args.break_step = int(break_step)  # [fix] the kit shipped int(32) with "TODO reset to 32e4"
    args.buffer_size = int(max_step * 32)
    args.repeat_times = 2
    args.horizon_len = int(max_step * 4)
    args.eval_per_step = int(max_step)
    args.num_workers = 1
    args.save_gap = 8

    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = dict(env_args, split="val")  # [fix] select on validation, not on the training rows

    ensemble_env = Ensemble(
        log_rules,
        save_path,
        1e6,
        agent_list,
        args,
    )
    ensemble_env.ensemble_train()
    return ensemble_env


if __name__ == "__main__":

    run(
        "ensemble_teamname",
        [AgentD3QN, AgentDoubleDQN, AgentDoubleDQN, AgentTwinD3QN],
    )
