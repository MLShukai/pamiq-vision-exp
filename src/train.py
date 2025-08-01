import logging
from datetime import datetime

import aim
import hydra
import rootutils
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


@hydra.main("./configs/train", "train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_view = cfg.copy()
    OmegaConf.resolve(cfg_view)
    logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg_view)}")

    # Convert device and dtype string object to pytorch object.
    shared_cfg = cfg.shared
    shared_cfg.device = f"${{torch.device:{shared_cfg.device}}}"
    shared_cfg.dtype = f"${{torch.dtype:{shared_cfg.dtype}}}"

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
        log_config(cfg_view, aim_run)

        launch(
            interaction=instantiate_interaction(cfg),
            models=instantiate_models(cfg),
            buffers=instantiate_buffers(cfg),
            trainers=instantiate_trainers(cfg),
            config=hydra.utils.instantiate(cfg.launch),
        )

    finally:
        aim_run.close()


def log_config(cfg: DictConfig, run: aim.Run) -> None:
    # Log flattened parameters
    log_targets = ["interaction", "models", "trainers", "buffers"]
    params = flatten_config({key: cfg[key] for key in log_targets})
    for key, value in params.items():
        run[key] = value


if __name__ == "__main__":
    main()
