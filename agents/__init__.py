from . import ppo_rnd
from . import ppo_replica
from . import ppo_worlds
from . import ppo_icm
from . import ppo_prediction_error
from . import ppo_episodic_curiosity
from . import ppo_disagreement
from . import ppo_icm_simple
from . import ppo_ride
from . import ppo_never_give_up

AGENTS = {
    "PPORND": ppo_rnd.PPORND,
    "PPO": ppo_replica.PPO,
    "PPOWorlds": ppo_worlds.PPOWorlds,
    "PPOIcm": ppo_icm.PPOIcm,
    "PPOPE": ppo_prediction_error.PPOPE,
    "PPOEpisodicCuriosity": ppo_episodic_curiosity.PPOEpisodicCuriosity,
    "PPODisagreement": ppo_disagreement.PPODisagreement,
    "PPOIcmSimple": ppo_icm_simple.PPOIcmSimple,
    "PPORide": ppo_ride.PPORide,
    "PPONeverGiveUp": ppo_never_give_up.PPONeverGiveUp,
}


def get_agent(cfg, envs, acmodel, agent_data, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in AGENTS,\
        "Please provide a valid Agent name."
    return AGENTS[cfg.name](cfg, envs, acmodel, agent_data, **kwargs)
