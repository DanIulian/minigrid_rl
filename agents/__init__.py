from . import ppo_rnd
from . import ppo_replica

AGENTS = {
    "PPORND": ppo_rnd.PPORND,
    "PPO": ppo_replica.PPO,
}


def get_agent(cfg, envs, acmodel, agent_data, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in AGENTS,\
        "Please provide a valid Agent name."
    return AGENTS[cfg.name](cfg, envs, acmodel, agent_data, **kwargs)
