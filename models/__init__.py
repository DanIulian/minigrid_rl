from . import model_rnd_like
from . import model_replica

MODELS = {
    "RNDModel": model_rnd_like.RNDModels,
    "Model": model_replica.Model,
}


def get_model(cfg, obs_space, action_space, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)
