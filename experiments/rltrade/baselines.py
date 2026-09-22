"""
Non-learning reference policies (brief section 3.2). They trade through the same
environment as the agents, so costs, execution lag and accounting are identical.

    buy_and_hold   enter long at the first bar (paying the entry cost), then hold
    random         uniform random action every decision, seeded
"""
import torch as th


def buy_and_hold_policy(obs):
    """Increase the level until fully long, then keep it (action 2 = +1, 1 = keep)."""
    full = obs[:, 0] >= 1.0 - 1e-6
    return th.where(full, th.ones_like(full, dtype=th.long), th.full_like(full, 2, dtype=th.long))


def random_policy(seed, action_dim=3):
    gen = th.Generator(device="cpu").manual_seed(seed)

    def policy(obs):
        return th.randint(0, action_dim, (obs.shape[0],), generator=gen).to(obs.device)
    return policy


BASELINES = ("buy_and_hold", "random")


def make_baseline(name, seed):
    if name == "buy_and_hold":
        return buy_and_hold_policy
    if name == "random":
        return random_policy(seed)
    raise ValueError(f"unknown baseline {name!r}")
