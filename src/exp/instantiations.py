import logging
from typing import Any

import hydra
import torch.nn as nn
from omegaconf import DictConfig
from pamiq_core import DataBuffer, Interaction
from pamiq_core.torch import TorchTrainer, TorchTrainingModel

from .utils import get_class_module_path

logger = logging.getLogger(__name__)


def instantiate_interaction(cfg: DictConfig) -> Interaction[Any, Any]:
    logger.info("Instantiating Interaction...")
    return hydra.utils.instantiate(cfg.interaction)


def instantiate_models(cfg: DictConfig) -> dict[str, TorchTrainingModel[Any]]:
    logger.info("Instantiating Models...")
    device, dtype = cfg.shared.device, cfg.shared.dtype

    models: dict[str, nn.Module] = hydra.utils.instantiate(cfg.models)
    for k, v in models.items():
        logger.info(
            f"Instantiated Model '{k}' at <{get_class_module_path(v.__class__)}>"
        )

    return {
        k: TorchTrainingModel(v, has_inference_model=False, device=device, dtype=dtype)
        for k, v in models.items()
    }


def instantiate_trainers(cfg: DictConfig) -> dict[str, TorchTrainer]:
    logger.info("Instantiating Trainers...")

    trainers_dict: dict[str, TorchTrainer] = hydra.utils.instantiate(cfg.trainers)

    for k, v in trainers_dict.items():
        logger.info(
            f"Instantiated Trainer '{k}' at <{get_class_module_path(v.__class__)}>"
        )

    return trainers_dict


def instantiate_buffers(cfg: DictConfig) -> dict[str, DataBuffer[Any, Any]]:
    logger.info("Instantiating DataBuffers...")
    buffers_dict: dict[str, DataBuffer[Any, Any]] = hydra.utils.instantiate(cfg.buffers)
    for k, v in buffers_dict.items():
        logger.info(
            f"Instantiated Buffer '{k}' at <{get_class_module_path(v.__class__)}>"
        )
    return buffers_dict
