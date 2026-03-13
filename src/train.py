"""Training entry point for vision representation learning."""

import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from exp.oc_resolvers import register_custom_resolvers
from exp.tracking import ExperimentTracker
from exp.training.checkpoint import CheckpointManager
from exp.training.loop import TrainingLoop

register_custom_resolvers()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run training pipeline."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # Create models via instantiate (returns dict[str, nn.Module])
    models: dict[str, nn.Module] = instantiate(cfg.model)

    # Move all models to device
    for name in models:
        models[name] = models[name].to(device)

    # Create data pipeline via instantiate
    frame_loader = instantiate(cfg.data.frame_loader)
    frame_stacker = instantiate(cfg.data.frame_stacker)
    buffer = instantiate(cfg.data.buffer)

    # Create training logic via instantiate (method-specific)
    training_logic = instantiate(cfg.training.training_logic, **models)

    tracker = ExperimentTracker(
        project_name=cfg.get("clearml", {}).get("project_name", ""),
        task_name=cfg.get("clearml", {}).get("task_name", cfg.experiment_name),
    )

    checkpoint_manager = CheckpointManager(
        log_dir=Path(cfg.log_dir),
        experiment_name=cfg.experiment_name,
    )

    training_loop = TrainingLoop(
        frame_loader=frame_loader,
        frame_stacker=frame_stacker,
        buffer=buffer,
        training_logic=training_logic,
        checkpoint_manager=checkpoint_manager,
        trigger_every_n_frames=cfg.training.trigger_every_n_frames,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        checkpoint_interval_seconds=cfg.training.checkpoint_interval_seconds,
        device=device,
        tracker=tracker,
    )

    training_loop.run(models)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
