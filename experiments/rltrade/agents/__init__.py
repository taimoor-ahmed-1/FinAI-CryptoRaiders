"""Agent registry. Every agent exposes the same interface:

    agent = make_agent(name, state_dim, action_dim, device, seed, **hyperparameters)
    agent.learn(env, total_steps, callback)      callback(agent, steps_done) -> True to stop
    agent.act(obs, deterministic=True)           LongTensor of actions
    agent.state_dict() / agent.load_state_dict(sd)
"""
from .ppo import PPO

AGENTS = {
    "ppo": PPO,
}


def make_agent(name, state_dim, action_dim, device, seed=0, **hp):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {tuple(AGENTS)}")
    return AGENTS[name](state_dim, action_dim, device=device, seed=seed, **hp)
