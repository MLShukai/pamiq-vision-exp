import logging
from pathlib import Path

import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf

from exp.oc_resolvers import register_custom_resolvers

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
    logger.info(f"Loaded exp configuration:\n{OmegaConf.to_yaml(exp_cfg)}")

    load_models(exp_cfg, state_path, cfg.shared.device, cfg.shared.dtype)


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
        logger.info(f"Loaded '{name}' model.")
        models[name] = m

    return models


if __name__ == "__main__":
    main()
