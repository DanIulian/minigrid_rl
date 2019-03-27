from . import ppo_rnd
from . import ppo_replica
from . import ppo_worlds
from . import ppo_worlds_separate
from . import ppo_icm

AGENTS = {
    "PPORND": ppo_rnd.PPORND,
    "PPO": ppo_replica.PPO,
    "PPOWorlds": ppo_worlds.PPOWorlds,
    "PPOWorldsSeparate": ppo_worlds_separate.PPOWorlds,
    "PPOIcm": ppo_icm.PPOIcm,

}


def get_agent(cfg, envs, acmodel, agent_data, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in AGENTS,\
        "Please provide a valid Agent name."
    return AGENTS[cfg.name](cfg, envs, acmodel, agent_data, **kwargs)
