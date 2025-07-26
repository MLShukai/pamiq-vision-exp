import logging
from datetime import datetime

import aim
import hydra
import rootutils
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pamiq_core import launch
from torch.utils.data import DataLoader

from exp.aim_utils import get_global_run, set_global_run
from exp.instantiations import (
    instantiate_buffers,
    instantiate_interaction,
    instantiate_models,
    instantiate_trainers,
)
from exp.models.names import ModelName
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
    # create pretrained model
    pretrained_model: nn.Module = hydra.utils.instantiate(cfg.models)[
        ModelName.JEPA_TARGET_ENCODER
    ]
    pretrained_model.load_state_dict(
        torch.load(cfg.pretrained_weight, weights_only=True)
    )
    pretrained_model = pretrained_model.to(cfg.device)
    # define linear probing head
    classifier = nn.Linear(cfg.models.embed_dim, 100, bias=True)
    classifier = classifier.to(cfg.device)

    optimizer = hydra.utils.instantiate(cfg.partial_optimizer)(classifier.parameters())
    dataloader = DataLoader(
        dataset=hydra.utils.instantiate(cfg.datasets.cifer100),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    global_step = 0
    for _ in range(cfg.max_epochs):
        images: torch.Tensor  # [batch_size, channels, height, width]
        labels: torch.Tensor  # [batch_size]
        for images, labels in dataloader:
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)
            with torch.no_grad():
                features = pretrained_model(
                    images
                )  # [batch_size, n_patches, embed_dim]
                features = features.mean(dim=1)  # [batch_size, embed_dim]
            logits = classifier(features)  # [batch_size, n_classes]
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging
            run.track(
                loss.item(),
                name="loss",
                step=global_step,
                context={"linear-probing": "jepa"},
            )
            global_step += 1


if __name__ == "__main__":
    main()
