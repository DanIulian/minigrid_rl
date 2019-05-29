from . import model_rnd_like
from . import model_replica
from . import model_worlds
from . import model_worlds_separate
from . import model_icm_like
from . import model_prediction_error

MODELS = {
    "RNDModel": model_rnd_like.RNDModels,
    "Model": model_replica.Model,
    "WorldsModels": model_worlds.WorldsModels,
    "WorldsModelsSeparate": model_worlds_separate.WorldsModels,
    "ICMModel": model_icm_like.ICMModel,
    "PEModel": model_prediction_error.PEModel,
}


def get_model(cfg, obs_space, action_space, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)
