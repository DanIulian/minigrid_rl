from . import model_rnd_like
from . import model_replica
from . import model_worlds
from . import model_icm_like
from . import model_prediction_error
from . import model_episodic_curiosity
from . import model_disagreement
from . import model_icm_simple
from . import model_never_give_up

MODELS = {
    "RNDModel": model_rnd_like.RNDModels,
    "Model": model_replica.Model,
    "WorldsModels": model_worlds.WorldsModels,
    "ICMModel": model_icm_like.ICMModel,
    "PEModel": model_prediction_error.PEModel,
    "EpisodicCuriosityModel": model_episodic_curiosity.EpisodicCuriosityModel,
    "DisagreementModel": model_disagreement.DisagreementModel,
    "ICMSimpleModel": model_icm_simple.ICMSimpleModel,
    "NeverGiveUpModel": model_never_give_up.NeverGiveUpModel,
}


def get_model(cfg, obs_space, action_space, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)
