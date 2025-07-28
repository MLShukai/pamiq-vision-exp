"""Static metrics computation for trained models.

This script loads a saved model state and computes metrics on a dataset.
Useful for evaluating model performance after training is complete.

Example usage:
    python static_metrics.py saved_state=/path/to/state.pt dataset=cifar100 metrics=jepa
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import aim
import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset

from exp.aim_utils import set_global_run
from exp.oc_resolvers import register_custom_resolvers
from exp.utils import get_class_module_path
from exp.validation.metrics_loggers import MetricsLogger

register_custom_resolvers()

rootutils.setup_root(__file__, indicator="pyproject.toml")

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


@hydra.main("./configs/static_metrics", "static_metrics.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Compute metrics for a saved model state."""
    # Display resolved configuration
    cfg_view = cfg.copy()
    OmegaConf.resolve(cfg_view)
    logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg_view)}")

    # Load saved state from experiment
    state_path = Path(cfg.saved_state)
    if not state_path.exists():
        raise ValueError(f"Specified state path is not found. {state_path}")
    logger.info(f"State path: {state_path}")

    # Load original experiment configuration
    exp_dir = state_path.parent.parent
    exp_cfg = OmegaConf.load(exp_dir / ".hydra/config.yaml")
    exp_cfg.update(OmegaConf.load(exp_dir / ".hydra/hydra.yaml"))
    logger.info("Loaded exp configuration.")

    # Setup device and dtype configuration
    shared_cfg = cfg.shared
    shared_cfg.device = f"${{torch.device:{shared_cfg.device}}}"
    shared_cfg.dtype = f"${{torch.dtype:{shared_cfg.dtype}}}"
    # Match image settings from original experiment
    shared_cfg.image = exp_cfg.shared.image

    # Load trained models from state
    models = load_models(exp_cfg, state_path, cfg.shared.device, cfg.shared.dtype)

    # Create metrics logger (e.g., JEPAMetrics)
    metrics: MetricsLogger = hydra.utils.instantiate(cfg.metrics)
    logger.info(
        f"Instantiated metrics logger: {get_class_module_path(metrics.__class__)}"
    )

    # Create evaluation dataset
    dataset: Dataset[Any] = hydra.utils.instantiate(cfg.dataset)
    logger.info(f"Instantiated dataset: {get_class_module_path(dataset.__class__)}")

    # Attach everything to metrics logger
    metrics.attach_exp_cfg(exp_cfg)
    metrics.attach_models(models)
    metrics.attach_dataset(dataset)

    # Initialize Aim tracking for metrics
    aim_run = aim.Run(
        repo=cfg.paths.aim_dir,
        experiment=cfg.experiment_name,
    )
    aim_run.name = (
        f"{cfg.experiment_name} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}"
    )

    # Add tags for easy filtering in Aim UI
    if cfg.tags:
        for tag in cfg.tags:
            aim_run.add_tag(tag)

    set_global_run(aim_run)

    # Run metrics computation
    try:
        logger.info("Starting metrics computation...")
        metrics.run()
        logger.info("Metrics computation completed successfully.")
    finally:
        aim_run.close()


def load_models(
    exp_cfg: DictConfig | ListConfig,
    state_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, nn.Module]:
    """Load models from saved state and prepare for evaluation."""
    # Instantiate models using original experiment config
    models: dict[str, nn.Module] = hydra.utils.instantiate(exp_cfg.models)

    # Load each model's state and set to evaluation mode
    for name, m in models.items():
        m = m.to(device, dtype)
        m.load_state_dict(
            torch.load(
                (state_path / "models" / name).with_suffix(".pt"), map_location=device
            )
        )
        m.eval()  # Important: set to evaluation mode
        logger.info(f"Loaded '{name}' model.")
        models[name] = m

    return models


if __name__ == "__main__":
    main()
