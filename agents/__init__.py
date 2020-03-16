from . import ppo_rnd
from . import ppo_replica
from . import ppo_custom_eval
from . import ppo_worlds
from . import ppo_worlds_separate
from . import ppo_icm
from . import ppo_prediction_error
from . import ppo_theorder
from . import ppo_conditioned

AGENTS = {
    "PPORND": ppo_rnd.PPORND,
    "PPO": ppo_replica.PPO,
    "PPOCustomEval": ppo_custom_eval.PPOCustomEval,
    "PPOConditioned": ppo_conditioned.PPOConditioned,
    "PPOOrder": ppo_theorder.PPOOrder,
    "PPOWorlds": ppo_worlds.PPOWorlds,
    "PPOWorldsSeparate": ppo_worlds_separate.PPOWorlds,
    "PPOIcm": ppo_icm.PPOIcm,
    "PPOPE": ppo_prediction_error.PPOPE

}


def get_agent(cfg, envs, acmodel, agent_data, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in AGENTS,\
        "Please provide a valid Agent name."
    return AGENTS[cfg.name](cfg, envs, acmodel, agent_data, **kwargs)
