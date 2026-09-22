"""
PPO for the discrete trading action space.

Defaults are the notebook's (5_erl_trainer, PPO_VERSION 5_7_3): clip 0.2, GAE
lambda 0.95, gamma 0.99, entropy 0.05, value coef 0.5, grad-norm 0.5, 6 epochs,
minibatch 64, 4096 transitions per update, lr 3e-4, target KL 0.02, separate
actor and critic with hidden sizes [256, 256, 128, 128, 64].

Fixed relative to the notebook (see CHANGES.md):
    - actions were SAMPLED at temperature 1.2 but their log-probabilities were
      re-evaluated at temperature 1 in the update, so the PPO ratio was wrong from
      the first epoch; sampling and update now use the same distribution;
    - rollouts come from all parallel environments, each acting on its own state
      (the notebook used one state and copied the action to every copy);
    - episode ends are time limits, so the value of the final state is bootstrapped
      instead of being treated as 0;
    - critic targets are the GAE returns (the notebook normalised the targets but
      not the value estimates they are compared with);
    - dropout off during rollouts and updates (dropout made the policy that acted
      differ from the one being updated). Available via `dropout=`.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch as th
import torch.nn.functional as F

from .nets import mlp


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 6
    minibatch: int = 64
    num_envs: int = 64
    rollout_len: int = 64              # 64 envs x 64 steps = 4096 transitions per update
    target_kl: float = 0.02
    net_dims: tuple = (256, 256, 128, 128, 64)
    dropout: float = 0.0
    adv_norm: bool = True


class PPO:
    name = "ppo"
    Config = PPOConfig

    def __init__(self, state_dim, action_dim, device="cpu", seed=0, **hp):
        self.cfg = self.Config(**hp)
        c = self.cfg
        self.device = th.device(device)
        th.manual_seed(seed)
        self.actor = mlp(state_dim, c.net_dims, action_dim, dropout=c.dropout, out_gain=0.01).to(self.device)
        self.critic = mlp(state_dim, c.net_dims, 1, dropout=c.dropout, out_gain=1.0).to(self.device)
        self.opt = th.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=c.lr, eps=1e-5)
        self.history = []

    # -------------------------------------------------------------- acting
    @th.no_grad()
    def act(self, obs, deterministic=True):
        self.actor.eval()
        logits = self.actor(obs.to(self.device))
        if deterministic:
            return logits.argmax(-1)
        return th.distributions.Categorical(logits=logits).sample()

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    # -------------------------------------------------------------- learning
    def _loss(self, obs, act, logp_old, adv, ret):
        dist = th.distributions.Categorical(logits=self.actor(obs))
        logp = dist.log_prob(act)
        ratio = (logp - logp_old).exp()
        pg = -th.min(ratio * adv, ratio.clamp(1 - self.cfg.clip, 1 + self.cfg.clip) * adv).mean()
        v_loss = F.mse_loss(self.value(obs), ret)
        ent = dist.entropy().mean()
        with th.no_grad():
            kl = ((ratio - 1) - (logp - logp_old)).mean()
        return pg + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * ent, v_loss, ent, kl

    def _update(self, b):
        c = self.cfg
        n = b["obs"].shape[0]
        mb = min(c.minibatch, n)
        stats = []
        for _ in range(c.epochs):
            perm = th.randperm(n, device=self.device)
            kls = []
            for i in range(0, n, mb):
                j = perm[i:i + mb]
                loss, v_loss, ent, kl = self._loss(b["obs"][j], b["act"][j], b["logp"][j], b["adv"][j], b["ret"][j])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                            c.max_grad_norm)
                self.opt.step()
                kls.append(float(kl))
                stats.append((float(v_loss), float(ent), float(kl)))
            if c.target_kl and np.mean(kls) > c.target_kl:        # notebook's early stop
                break
        s = np.asarray(stats)
        return {"value_loss": s[:, 0].mean(), "entropy": s[:, 1].mean(), "approx_kl": s[:, 2].mean()}

    def learn(self, env, total_steps, callback=None, callback_every=0):
        """Train on `env` (a TradingEnv with num_envs = cfg.num_envs) for `total_steps` transitions.

        `callback(agent, steps_done)` is called about every `callback_every` transitions
        (checkpointing on validation, pruning); returning True stops training.
        """
        c, dev = self.cfg, self.device
        N, T = env.n_env, c.rollout_len
        obs = env.reset().to(dev)
        steps, next_cb = 0, callback_every
        while steps < total_steps:
            self.actor.train(), self.critic.train()
            buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
            raw_rew = 0.0
            with th.no_grad():
                for _ in range(T):
                    logits = self.actor(obs)
                    dist = th.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    v = self.value(obs)
                    nobs, r, done, info = env.step(a)
                    r = r.to(dev)
                    raw_rew += float(r.mean())
                    if "final_obs" in info:                    # time limit: bootstrap V(s_T)
                        r = r + c.gamma * done.float() * self.value(info["final_obs"].to(dev))
                    for k, x in zip(buf, (obs, a, dist.log_prob(a), v, r, done.float())):
                        buf[k].append(x)
                    obs = nobs.to(dev)
                last_v = self.value(obs)
            b = {k: th.stack(v) for k, v in buf.items()}          # (T, N, ...)
            adv = th.zeros_like(b["rew"])
            gae = th.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else b["val"][t + 1]
                nonterm = 1.0 - b["done"][t]
                delta = b["rew"][t] + c.gamma * nv * nonterm - b["val"][t]
                gae = delta + c.gamma * c.gae_lambda * nonterm * gae
                adv[t] = gae
            ret = (adv + b["val"]).reshape(-1)
            adv = adv.reshape(-1)
            if c.adv_norm:                                        # over the whole batch, as the notebook
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            flat = {"obs": b["obs"].reshape(T * N, -1), "act": b["act"].reshape(-1),
                    "logp": b["logp"].reshape(-1), "adv": adv, "ret": ret}
            st = self._update(flat)
            steps += T * N
            st.update(steps=steps, reward_mean=raw_rew / T)
            self.history.append(st)
            if callback is not None and steps >= next_cb:
                next_cb += callback_every
                if callback(self, steps):
                    break
        return self.history

    # -------------------------------------------------------------- persistence
    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])

    def config(self):
        return asdict(self.cfg)
