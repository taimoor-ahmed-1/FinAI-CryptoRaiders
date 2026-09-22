"""
Value-based agents for the discrete action space.

    dqn    DQN (Mnih et al., 2015): replay buffer, target network, epsilon-greedy
    ddqn   Double DQN (van Hasselt et al., 2016): the online net picks the next
           action, the target net evaluates it - removes max-operator over-estimation
    d3qn   Dueling Double DQN (Wang et al., 2016): Q = V + (A - mean A)

Episode ends in the trading environment are time limits, not terminal states, so
every transition bootstraps from the next state (`final_obs` at an episode end).
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nets import mlp


@dataclass
class DQNConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 500_000
    batch_size: int = 256
    learning_starts: int = 20_000       # transitions of uniform-random play before learning
    replay_ratio: float = 0.25          # gradient steps per collected transition
    target_update: int = 2_000          # gradient steps between hard target-network copies
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3           # share of the budget over which epsilon decays
    num_envs: int = 16
    net_dims: tuple = (256, 256, 128)
    max_grad_norm: float = 10.0
    double: bool = False
    dueling: bool = False


class DuelingNet(nn.Module):
    def __init__(self, state_dim, action_dim, net_dims):
        super().__init__()
        self.body = mlp(state_dim, net_dims[:-1], net_dims[-1])
        self.v = mlp(net_dims[-1], (), 1)
        self.a = mlp(net_dims[-1], (), action_dim, out_gain=0.01)

    def forward(self, x):
        h = F.relu(self.body(x))
        a = self.a(h)
        return self.v(h) + a - a.mean(-1, keepdim=True)


class Replay:
    def __init__(self, size, state_dim, device):
        self.size, self.ptr, self.full, self.dev = size, 0, False, device
        self.obs = th.zeros((size, state_dim), device=device)
        self.nobs = th.zeros((size, state_dim), device=device)
        self.act = th.zeros(size, dtype=th.long, device=device)
        self.rew = th.zeros(size, device=device)

    def add(self, obs, act, rew, nobs):
        k = obs.shape[0]
        idx = (th.arange(k, device=self.dev) + self.ptr) % self.size
        self.obs[idx], self.act[idx], self.rew[idx], self.nobs[idx] = obs, act, rew, nobs
        self.ptr = (self.ptr + k) % self.size
        self.full = self.full or self.ptr < k

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n, gen):
        i = th.randint(0, len(self), (n,), generator=gen).to(self.dev)
        return self.obs[i], self.act[i], self.rew[i], self.nobs[i]


class DQN:
    name = "dqn"
    Config = DQNConfig
    defaults = {}

    def __init__(self, state_dim, action_dim, device="cpu", seed=0, **hp):
        self.cfg = self.Config(**{**self.defaults, **hp})
        c = self.cfg
        self.device = th.device(device)
        th.manual_seed(seed)
        self.gen = th.Generator(device="cpu").manual_seed(seed)
        self.action_dim = action_dim
        make = (lambda: DuelingNet(state_dim, action_dim, c.net_dims)) if c.dueling else \
               (lambda: mlp(state_dim, c.net_dims, action_dim))
        self.q, self.q_target = make().to(self.device), make().to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = th.optim.Adam(self.q.parameters(), lr=c.lr)
        self.replay = Replay(c.buffer_size, state_dim, self.device)
        self.history, self.grad_steps = [], 0

    @th.no_grad()
    def act(self, obs, deterministic=True, eps=0.0):
        obs = obs.to(self.device)
        greedy = self.q(obs).argmax(-1)
        if deterministic or eps <= 0:
            return greedy
        rand = th.randint(0, self.action_dim, greedy.shape, generator=self.gen).to(self.device)
        explore = th.rand(greedy.shape, generator=self.gen).to(self.device) < eps
        return th.where(explore, rand, greedy)

    def _train_step(self):
        c = self.cfg
        obs, act, rew, nobs = self.replay.sample(c.batch_size, self.gen)
        with th.no_grad():
            if c.double:
                a_next = self.q(nobs).argmax(-1, keepdim=True)
                q_next = self.q_target(nobs).gather(1, a_next).squeeze(1)
            else:
                q_next = self.q_target(nobs).max(-1).values
            target = rew + c.gamma * q_next
        q = self.q(obs).gather(1, act[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.q.parameters(), c.max_grad_norm)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % c.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss), float(q.mean())

    def learn(self, env, total_steps, callback=None, callback_every=0):
        c, dev = self.cfg, self.device
        obs = env.reset().to(dev)
        steps, next_cb, owed = 0, callback_every, 0.0
        losses, qs, rews = [], [], []
        while steps < total_steps:
            self.q.train()
            frac = min(1.0, steps / max(1.0, c.eps_fraction * total_steps))
            eps = c.eps_start + frac * (c.eps_end - c.eps_start)
            if steps < c.learning_starts:
                a = th.randint(0, self.action_dim, (env.n_env,), generator=self.gen).to(dev)
            else:
                a = self.act(obs, deterministic=False, eps=eps)
            nobs, r, done, info = env.step(a)
            nobs, r = nobs.to(dev), r.to(dev)
            target_obs = nobs
            if "final_obs" in info:
                target_obs = th.where(done.to(dev)[:, None], info["final_obs"].to(dev), nobs)
            self.replay.add(obs, a, r, target_obs)
            obs = nobs
            steps += env.n_env
            rews.append(float(r.mean()))
            if steps >= c.learning_starts:
                owed += env.n_env * c.replay_ratio
                while owed >= 1.0:
                    l, q = self._train_step()
                    losses.append(l), qs.append(q)
                    owed -= 1.0
            if len(rews) >= 256:
                self.history.append({"steps": steps, "eps": eps, "reward_mean": float(np.mean(rews)),
                                     "loss": float(np.mean(losses)) if losses else float("nan"),
                                     "q_mean": float(np.mean(qs)) if qs else float("nan")})
                losses, qs, rews = [], [], []
            if callback is not None and steps >= next_cb:
                next_cb += callback_every
                if callback(self, steps):
                    break
        return self.history

    def state_dict(self):
        return {"q": self.q.state_dict(), "q_target": self.q_target.state_dict()}

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q_target"])

    def config(self):
        return asdict(self.cfg)


class DoubleDQN(DQN):
    name = "ddqn"
    defaults = {"double": True}


class D3QN(DQN):
    name = "d3qn"
    defaults = {"double": True, "dueling": True}
