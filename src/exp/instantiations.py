import logging
from typing import Any

import hydra
import torch.nn as nn
from omegaconf import DictConfig
from pamiq_core import Interaction
from pamiq_core.torch import TorchTrainingModel

logger = logging.getLogger(__name__)


def instantiate_interaction(cfg: DictConfig) -> Interaction[Any, Any]:
    logger.info("Instantiating Interaction...")
    return hydra.utils.instantiate(cfg.interaction)


def instantiate_models(cfg: DictConfig) -> dict[str, TorchTrainingModel[Any]]:
    logger.info("Instantiating Models...")
    device, dtype = cfg.shared.device, cfg.shared.dtype

    models_cfg = cfg.models

    models: dict[str, nn.Module]
    if "_target_" in models_cfg:
        logger.info(f"Instantiating models: <{models_cfg._target_}>")
        models = hydra.utils.instantiate(models_cfg)
    else:
        models = {}
        for k, v in models_cfg.items():
            logger.info(f"Instantiating model '{k}': <{v._target_}>")
            models[str(k)] = hydra.utils.instantiate(v)

    return {
        k: TorchTrainingModel(v, has_inference_model=False, device=device, dtype=dtype)
        for k, v in models.items()
    }
