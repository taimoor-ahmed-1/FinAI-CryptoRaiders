"""
Reward functions. Each is a small stateful object evaluated for all parallel
environments at once:

    r = reward(eq_prev, eq_next, level_prev, level_new)      all tensors of shape (num_envs,)
    reward.reset(mask)                                        clear state of envs that restarted

Equity starts every episode at 1.0, so `eq_next - eq_prev` is the step's profit
as a fraction of starting capital, net of transaction costs (costs are already
deducted from `eq_next`).

    net_asset_change   change in net assets - the notebook's reward, now relative
                       to starting capital (default)
    legacy_shaped      net_asset_change plus the notebook's two shaping terms:
                       +0.001 per unit of position change and +1% of any positive
                       reward. Kept only to show what the shaping did; it pays
                       the agent for trading, which fights the transaction costs.
"""
import torch as th


class Reward:
    def __init__(self, num_envs, device, scale=100.0):
        self.n, self.device, self.scale = num_envs, device, scale

    def reset(self, mask=None):
        pass

    def __call__(self, eq_prev, eq_next, level_prev, level_new):
        raise NotImplementedError


class NetAssetChange(Reward):
    def __call__(self, eq_prev, eq_next, level_prev, level_new):
        return (eq_next - eq_prev) * self.scale


class LegacyShaped(Reward):
    def __init__(self, num_envs, device, scale=100.0, trade_bonus=0.001, profit_bonus=0.01):
        super().__init__(num_envs, device, scale)
        self.trade_bonus, self.profit_bonus = trade_bonus, profit_bonus

    def __call__(self, eq_prev, eq_next, level_prev, level_new):
        base = (eq_next - eq_prev) * self.scale
        return (base + self.trade_bonus * (level_new - level_prev).abs()
                + self.profit_bonus * base.clamp(min=0))


REWARDS = {
    "net_asset_change": NetAssetChange,
    "legacy_shaped": LegacyShaped,
}


def make_reward(name, num_envs, device, scale=None, **kwargs):
    """`scale=None` keeps the reward's own default."""
    if name not in REWARDS:
        raise ValueError(f"unknown reward {name!r}; choose from {tuple(REWARDS)}")
    if scale is not None:
        kwargs["scale"] = scale
    return REWARDS[name](num_envs, device, **kwargs)
