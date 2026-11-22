"""
Soft Actor-Critic for the discrete trading action space (experiment 22).

SAC-Discrete (Christodoulou, 2019): the actor outputs a categorical policy over the 3 actions,
two critics output Q(s, .) for every action, and because the action set is small every
expectation over actions is computed exactly instead of sampled:

    V(s')       = sum_a pi(a|s') [ min(Q1', Q2')(s', a) - alpha * log pi(a|s') ]
    critic loss = Huber(Q_i(s, a), r + gamma * V(s'))           for i = 1, 2
    actor loss  = sum_a pi(a|s) [ alpha * log pi(a|s) - min(Q1, Q2)(s, a) ]
    alpha loss  = log(alpha) * (H(pi(.|s)) - H_target)          H_target = ratio * log(3)

Target critics follow the online critics by Polyak averaging (tau). The Huber loss (as in
our DQN) keeps the large reward scales of the search (up to 1000x) from blowing up the
critic. Episode ends are time limits, so every transition bootstraps from the next state
(`final_obs` at an episode end), as in dqn.py. The replay buffer is dqn.Replay.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch as th
import torch.nn.functional as F

from .dqn import Replay
from .nets import mlp


@dataclass
class SACConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005                  # Polyak rate of the target critics, per gradient step
    buffer_size: int = 500_000
    batch_size: int = 256
    learning_starts: int = 20_000       # transitions of uniform-random play before learning
    replay_ratio: float = 0.25          # gradient steps per collected transition
    target_entropy_ratio: float = 0.5   # entropy target as a share of the maximum, log(3)
    init_alpha: float = 0.2
    num_envs: int = 16
    net_dims: tuple = (256, 256, 128)
    max_grad_norm: float = 10.0


class SAC:
    name = "sac"
    Config = SACConfig

    def __init__(self, state_dim, action_dim, device="cpu", seed=0, **hp):
        self.cfg = self.Config(**hp)
        c = self.cfg
        self.device = th.device(device)
        th.manual_seed(seed)
        self.gen = th.Generator(device="cpu").manual_seed(seed)
        self.action_dim = action_dim
        self.actor = mlp(state_dim, c.net_dims, action_dim, out_gain=0.01).to(self.device)
        self.q1, self.q2 = (mlp(state_dim, c.net_dims, action_dim).to(self.device) for _ in range(2))
        self.q1_t, self.q2_t = (mlp(state_dim, c.net_dims, action_dim).to(self.device) for _ in range(2))
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        self.log_alpha = th.tensor(math.log(c.init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = c.target_entropy_ratio * math.log(action_dim)
        self.opt_actor = th.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.opt_q = th.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=c.lr)
        self.opt_alpha = th.optim.Adam([self.log_alpha], lr=c.lr)
        self.replay = Replay(c.buffer_size, state_dim, self.device)
        self.history, self.grad_steps = [], 0

    @property
    def alpha(self):
        return self.log_alpha.detach().exp()

    @th.no_grad()
    def act(self, obs, deterministic=True):
        logits = self.actor(obs.to(self.device))
        if deterministic:
            return logits.argmax(-1)
        u = th.rand(logits.shape, generator=self.gen).to(self.device)        # Gumbel-max sample, seeded
        return (logits - th.log(-th.log(u.clamp_min(1e-12)))).argmax(-1)

    def _train_step(self):
        c = self.cfg
        obs, act, rew, nobs = self.replay.sample(c.batch_size, self.gen)
        alpha = self.alpha
        with th.no_grad():
            logp_n = F.log_softmax(self.actor(nobs), -1)
            q_n = th.min(self.q1_t(nobs), self.q2_t(nobs))
            v_n = (logp_n.exp() * (q_n - alpha * logp_n)).sum(-1)
            target = rew + c.gamma * v_n
        q1 = self.q1(obs).gather(1, act[:, None]).squeeze(1)
        q2 = self.q2(obs).gather(1, act[:, None]).squeeze(1)
        q_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        th.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), c.max_grad_norm)
        self.opt_q.step()

        logp = F.log_softmax(self.actor(obs), -1)
        p = logp.exp()
        with th.no_grad():
            q_min = th.min(self.q1(obs), self.q2(obs))
        actor_loss = (p * (alpha * logp - q_min)).sum(-1).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
        self.opt_actor.step()

        entropy = -(p * logp).sum(-1).mean().detach()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        with th.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for a, b in zip(net.parameters(), tgt.parameters()):
                    b.mul_(1 - c.tau).add_(a, alpha=c.tau)
        self.grad_steps += 1
        return float(q_loss.detach()), float(q1.detach().mean()), float(entropy), float(alpha)

    def learn(self, env, total_steps, callback=None, callback_every=0):
        c, dev = self.cfg, self.device
        obs = env.reset().to(dev)
        steps, next_cb, owed = 0, callback_every, 0.0
        stats, rews = [], []
        while steps < total_steps:
            if steps < c.learning_starts:
                a = th.randint(0, self.action_dim, (env.n_env,), generator=self.gen).to(dev)
            else:
                a = self.act(obs, deterministic=False)
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
                    stats.append(self._train_step())
                    owed -= 1.0
            if len(rews) >= 256:
                s = np.asarray(stats) if stats else np.full((1, 4), np.nan)
                self.history.append({"steps": steps, "reward_mean": float(np.mean(rews)),
                                     "loss": float(s[:, 0].mean()), "q_mean": float(s[:, 1].mean()),
                                     "entropy": float(s[:, 2].mean()), "alpha": float(s[:, 3].mean())})
                stats, rews = [], []
            if callback is not None and steps >= next_cb:
                next_cb += callback_every
                if callback(self, steps):
                    break
        return self.history

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "q1_t": self.q1_t.state_dict(), "q2_t": self.q2_t.state_dict(),
                "alpha": {"log_alpha": self.log_alpha.detach().clone()}}

    def load_state_dict(self, sd):
        for k in ("actor", "q1", "q2", "q1_t", "q2_t"):
            getattr(self, k).load_state_dict(sd[k])
        with th.no_grad():
            self.log_alpha.copy_(sd["alpha"]["log_alpha"].to(self.device))

    def config(self):
        return asdict(self.cfg)
