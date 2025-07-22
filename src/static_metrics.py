import logging
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

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

    exp_cfg = OmegaConf.load(f"{cfg.exp_dir}/.hydra/config.yaml")
    exp_cfg.update(OmegaConf.load(f"{cfg.exp_dir}/.hydra/hydra.yaml"))
    logger.info(f"Loaded exp configuration:\n{OmegaConf.to_yaml(exp_cfg)}")

    state_path = Path(f"{cfg.exp_dir}/states/{cfg.state}")
    if not state_path.exists():
        raise ValueError(f"Specified state path is not found. {state_path}")
    logger.info(f"State path: {state_path}")


if __name__ == "__main__":
    main()
