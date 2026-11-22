"""Agent registry. Every agent exposes the same interface:

    agent = make_agent(name, state_dim, action_dim, device, seed, **hyperparameters)
    agent.learn(env, total_steps, callback)      callback(agent, steps_done) -> True to stop
    agent.act(obs, deterministic=True)           LongTensor of actions
    agent.policy(deterministic=True)             optional: a stateful policy (recurrent agents), reset per backtest
    agent.state_dict() / agent.load_state_dict(sd)
"""
from .a2c import A2C
from .dqn import D3QN, DQN, DoubleDQN
from .ppo import PPO
from .rppo import RecurrentPPOAgent
from .sac import SAC

AGENTS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "ddqn": DoubleDQN,
    "d3qn": D3QN,
    "sac": SAC,                      # discrete SAC (experiment 22)
    "rppo": RecurrentPPOAgent,       # sb3-contrib RecurrentPPO, LSTM (experiment 22)
}


def make_agent(name, state_dim, action_dim, device, seed=0, **hp):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; choose from {tuple(AGENTS)}")
    return AGENTS[name](state_dim, action_dim, device=device, seed=seed, **hp)


def agent_policy(agent, deterministic=True):
    """policy(obs) -> actions for backtests. Recurrent agents return a stateful policy that
    env.run_policy resets at the start of every backtest; the others act from the observation."""
    if hasattr(agent, "policy"):
        return agent.policy(deterministic)
    return lambda o: agent.act(o, deterministic=deterministic)
