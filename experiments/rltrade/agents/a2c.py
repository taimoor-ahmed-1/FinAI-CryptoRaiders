"""
A2C (synchronous advantage actor-critic).

Same networks and rollout machinery as PPO; the differences are the update:
one gradient step per rollout on the whole batch, plain policy-gradient loss
(no ratio clipping, no KL stop), shorter rollouts and n-step returns
(gae_lambda = 1). Defaults follow common A2C practice (Mnih et al., 2016;
Stable-Baselines3), adjusted to the PPO network for a like-for-like comparison.
"""
from dataclasses import dataclass

import torch as th
import torch.nn.functional as F

from .ppo import PPO, PPOConfig


@dataclass
class A2CConfig(PPOConfig):
    lr: float = 7e-4
    gae_lambda: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 1
    minibatch: int = 1 << 30           # whole rollout in one step
    num_envs: int = 64
    rollout_len: int = 16
    target_kl: float = 0.0


class A2C(PPO):
    name = "a2c"
    Config = A2CConfig

    def _loss(self, obs, act, logp_old, adv, ret):
        dist = th.distributions.Categorical(logits=self.actor(obs))
        pg = -(dist.log_prob(act) * adv).mean()
        v_loss = F.mse_loss(self.value(obs), ret)
        ent = dist.entropy().mean()
        return pg + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * ent, v_loss, ent, th.zeros(())
