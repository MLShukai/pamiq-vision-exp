import hydra
import rootutils
from omegaconf import DictConfig
from pamiq_core import LaunchConfig, launch

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator="pyproject.toml", pythonpath=True
)


@hydra.main("./configs", "train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pass


if __name__ == "__main__":
    main()
