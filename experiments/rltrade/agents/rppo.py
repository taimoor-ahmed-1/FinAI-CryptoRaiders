"""
Recurrent PPO (an LSTM actor-critic) from sb3-contrib, behind the common agent interface (experiment 22).

sb3-contrib's `RecurrentPPO` (Raffin et al., Stable-Baselines3) is used unchanged. Two adapters
connect it to this code base:

    TorchVecEnv   presents our batched GPU TradingEnv as one Stable-Baselines3 VecEnv, so every
                  copy is stepped in a single call. Episode ends are time limits: each finished
                  copy reports `terminal_observation` and `TimeLimit.truncated`, so RecurrentPPO
                  bootstraps V(s_T) instead of treating the end as terminal (as our PPO does).
    policy()      an LSTM policy has a memory, so evaluation needs a fresh hidden state per
                  backtest. `policy()` returns a callable that keeps one hidden state per copy
                  and is reset by `env.run_policy` at the start of every backtest.

`act()` without a memory is deliberately not offered.

sb3-contrib, stable-baselines3 and gymnasium are imported only when this agent is built, so
the other agents and notebooks do not need them.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch as th


@dataclass
class RPPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch: int = 512                 # sb3 batch_size (transitions per gradient step)
    num_envs: int = 64
    rollout_len: int = 128               # sb3 n_steps per copy: 64 x 128 = 8192 transitions per update
    net_dims: tuple = (128, 128)         # MLP after the LSTM, for the actor and for the critic
    lstm_hidden: int = 128
    target_kl: float = 0.0               # 0 = no early stop


def _vec_env_class():
    import gymnasium as gym
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class SpacesOnlyVecEnv(VecEnv):
        """Carries the spaces and the number of copies; used to build or load a model without data."""

        def __init__(self, num_envs, state_dim):
            obs = gym.spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)
            super().__init__(num_envs, obs, gym.spaces.Discrete(3))

        def reset(self):
            raise RuntimeError("no data attached")

        def step_async(self, actions):
            raise RuntimeError("no data attached")

        def step_wait(self):
            raise RuntimeError("no data attached")

        def close(self):
            pass

        def get_attr(self, attr_name, indices=None):
            return [None] * self.num_envs

        def set_attr(self, attr_name, value, indices=None):
            pass

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return [None] * self.num_envs

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

    class TorchVecEnv(SpacesOnlyVecEnv):
        def __init__(self, env):
            super().__init__(env.n_env, env.state_dim)
            self.env, self._a = env, None

        def reset(self):
            return self.env.reset().float().cpu().numpy()

        def step_async(self, actions):
            self._a = th.as_tensor(np.asarray(actions).reshape(-1), dtype=th.long)

        def step_wait(self):
            obs, r, done, info = self.env.step(self._a)
            obs = obs.float().cpu().numpy()
            done = done.cpu().numpy().astype(bool)
            infos = [{} for _ in range(self.num_envs)]
            if done.any():
                fin = info["final_obs"].float().cpu().numpy()
                for i in np.flatnonzero(done):
                    infos[i] = {"terminal_observation": fin[i], "TimeLimit.truncated": True}
            return obs, r.float().cpu().numpy(), done, infos

    return SpacesOnlyVecEnv, TorchVecEnv


class LSTMPolicy:
    """A deterministic or sampled LSTM policy with one hidden state per environment copy."""

    def __init__(self, model, deterministic=True):
        self.model, self.deterministic = model, deterministic
        self.reset()

    def reset(self):
        self.state, self.start = None, None

    def __call__(self, obs):
        o = obs.float().cpu().numpy()
        if self.start is None or len(self.start) != len(o):
            self.state, self.start = None, np.ones(len(o), dtype=bool)
        a, self.state = self.model.predict(o, state=self.state, episode_start=self.start,
                                           deterministic=self.deterministic)
        self.start = np.zeros(len(o), dtype=bool)
        return th.as_tensor(a, dtype=th.long, device=obs.device)


class RecurrentPPOAgent:
    name = "rppo"
    Config = RPPOConfig

    def __init__(self, state_dim, action_dim, device="cpu", seed=0, **hp):
        from sb3_contrib import RecurrentPPO
        assert action_dim == 3, "TorchVecEnv exposes the 3 trading actions"
        self.cfg = self.Config(**hp)
        c = self.cfg
        self.device, self.seed, self.state_dim = th.device(device), seed, state_dim
        spaces_env, self._TorchVecEnv = _vec_env_class()
        th.manual_seed(seed)
        self.model = RecurrentPPO(
            "MlpLstmPolicy", spaces_env(c.num_envs, state_dim), learning_rate=c.lr, n_steps=c.rollout_len,
            batch_size=c.minibatch, n_epochs=c.epochs, gamma=c.gamma, gae_lambda=c.gae_lambda, clip_range=c.clip,
            ent_coef=c.entropy_coef, vf_coef=c.value_coef, max_grad_norm=c.max_grad_norm,
            target_kl=c.target_kl or None, seed=seed, device=str(self.device), verbose=0,
            policy_kwargs={"lstm_hidden_size": c.lstm_hidden,
                           "net_arch": {"pi": list(c.net_dims), "vf": list(c.net_dims)}})
        self.history = []

    def policy(self, deterministic=True):
        return LSTMPolicy(self.model, deterministic)

    def act(self, obs, deterministic=True):
        raise NotImplementedError("an LSTM policy needs a memory: use agent.policy() (runner.agent_policy)")

    def learn(self, env, total_steps, callback=None, callback_every=0):
        from stable_baselines3.common.callbacks import BaseCallback
        agent = self

        class Hook(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_cb = callback_every

            def _on_step(self):
                if callback is not None and callback_every and self.num_timesteps >= self.next_cb:
                    self.next_cb += callback_every
                    if callback(agent, self.num_timesteps):
                        return False
                return True

            def _on_rollout_end(self):
                buf = self.model.rollout_buffer
                agent.history.append({"steps": int(self.num_timesteps),
                                      "reward_mean": float(np.mean(buf.rewards)),
                                      "value_mean": float(np.mean(buf.values))})

        self.model.set_env(self._TorchVecEnv(env))
        self.model.learn(total_timesteps=total_steps, callback=Hook(), reset_num_timesteps=True)
        return self.history

    def state_dict(self):
        return {"policy": self.model.policy.state_dict()}

    def load_state_dict(self, sd):
        self.model.policy.load_state_dict({k: v.to(self.model.device) for k, v in sd["policy"].items()})

    def config(self):
        return asdict(self.cfg)
