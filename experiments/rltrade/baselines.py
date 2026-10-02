"""
Non-learning reference policies (brief section 3.2). They trade through the same
environment as the agents, so costs, execution lag and accounting are identical.

    buy_and_hold   enter long at the first bar (paying the entry cost), then hold
    random         uniform random action every decision, seeded
    cash           never trade (stay flat): the "do nothing" reference that RL
                   checkpoints are not allowed to win with (see runner.selection_score)
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


def cash_policy(obs):
    """Keep the (flat) starting position forever: action 1 = keep."""
    return th.ones(obs.shape[0], dtype=th.long, device=obs.device)


BASELINES = ("buy_and_hold", "random", "cash")


def make_baseline(name, seed):
    if name == "buy_and_hold":
        return buy_and_hold_policy
    if name == "random":
        return random_policy(seed)
    if name == "cash":
        return cash_policy
    raise ValueError(f"unknown baseline {name!r}")
