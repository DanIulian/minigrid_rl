from . import model_rnd_like
from . import model_replica
from . import model_aux_in
from . import model_worlds
from . import model_worlds_separate
from . import model_icm_like
from . import model_prediction_error
from . import model_theorder
from . import model_theorder_oracle
from . import model_conditioned

MODELS = {
    "OrderModels": model_theorder.OrderModels,
    "ModelConditioned": model_conditioned.ModelConditioned,
    "OrderModelsOracle": model_theorder_oracle.OrderModelsOracle,
    "RNDModel": model_rnd_like.RNDModels,
    "Model": model_replica.Model,
    "ModelAuxIn": model_aux_in.ModelAuxIn,
    "WorldsModels": model_worlds.WorldsModels,
    "WorldsModelsSeparate": model_worlds_separate.WorldsModels,
    "ICMModel": model_icm_like.ICMModel,
    "PEModel": model_prediction_error.PEModel,
}


def get_model(cfg, obs_space, action_space, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)
