import logging
from datetime import datetime

import aim
import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pamiq_core import launch

from exp.aim_utils import flatten_config, set_global_run
from exp.instantiations import (
    instantiate_buffers,
    instantiate_interaction,
    instantiate_models,
    instantiate_trainers,
)
from exp.oc_resolvers import register_custom_resolvers

register_custom_resolvers()

rootutils.setup_root(__file__, indicator="pyproject.toml")

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


@hydra.main("./configs/linear_probing/", "linear_probing.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_view = cfg.copy()
    OmegaConf.resolve(cfg_view)
    logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg_view)}")

    # Initialize Aim Run
    aim_run = aim.Run(
        repo=cfg.paths.aim_dir,
        experiment=cfg.experiment_name,
        system_tracking_interval=10,  # Track system metrics every 10 seconds
    )
    aim_run.name = (
        f"{cfg.experiment_name} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}"
    )

    # Set tags if available
    if cfg.tags:
        for tag in cfg.tags:
            aim_run.add_tag(tag)

    set_global_run(aim_run)

    try:
        train(cfg_view, aim_run)
    finally:
        aim_run.close()


def train(cfg: DictConfig, run: aim.Run) -> None:
    # target_pretrained_model: nn.Module = hydra.utils.instantiate(cfg.models)
    pass


if __name__ == "__main__":
    main()
