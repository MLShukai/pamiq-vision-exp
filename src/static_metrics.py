import logging
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import Dataset

from exp.oc_resolvers import register_custom_resolvers
from exp.utils import get_class_module_path
from exp.validation.metrics_loggers import MetricsLogger

register_custom_resolvers()

rootutils.setup_root(__file__, indicator="pyproject.toml")

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


@hydra.main("./configs/static_metrics", "static_metrics.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_view = cfg.copy()
    OmegaConf.resolve(cfg_view)
    logger.info(f"Loaded configuration:\n{OmegaConf.to_yaml(cfg_view)}")

    # Convert device and dtype string object to pytorch object.
    shared_cfg = cfg.shared
    shared_cfg.device = f"${{torch.device:{shared_cfg.device}}}"
    shared_cfg.dtype = f"${{torch.dtype:{shared_cfg.dtype}}}"

    state_path = Path(cfg.saved_state)
    if not state_path.exists():
        raise ValueError(f"Specified state path is not found. {state_path}")
    logger.info(f"State path: {state_path}")
    exp_dir = state_path.parent.parent

    exp_cfg = OmegaConf.load(exp_dir / ".hydra/config.yaml")
    exp_cfg.update(OmegaConf.load(exp_dir / ".hydra/hydra.yaml"))
    with open_dict(shared_cfg):
        shared_cfg.image = exp_cfg.shared.image
    logger.info("Loaded exp configuration.")

    models = load_models(exp_cfg, state_path, cfg.shared.device, cfg.shared.dtype)

    metrics: MetricsLogger = hydra.utils.instantiate(cfg.metrics)
    logger.info(
        f"Instantiated metrics logger: {get_class_module_path(metrics.__class__)}"
    )

    dataset: Dataset[Any] = hydra.utils.instantiate(cfg.dataset)
    logger.info(f"Instantiated dataset: {get_class_module_path(dataset.__class__)}")

    metrics.attach_exp_cfg(exp_cfg)
    metrics.attach_models(models)
    metrics.attach_dataset(dataset)

    metrics.run()


def load_models(
    exp_cfg: DictConfig | ListConfig,
    state_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = hydra.utils.instantiate(exp_cfg.models)

    for name, m in models.items():
        m = m.to(device, dtype)
        m.load_state_dict(
            torch.load(
                (state_path / "models" / name).with_suffix(".pt"), map_location=device
            )
        )
        m.eval()
        logger.info(f"Loaded '{name}' model.")
        models[name] = m

    return models


if __name__ == "__main__":
    main()
