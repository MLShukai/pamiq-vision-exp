"""Training entry point for vision representation learning."""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from exp.models.components.patchfier import VideoPatchifier
from exp.models.vjepa import Encoder, Predictor
from exp.tracking import ExperimentTracker
from exp.training.checkpoint import CheckpointManager
from exp.training.loop import TrainingLoop

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run training pipeline."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # Create model via instantiate (calls create_video_jepa)
    models = instantiate(cfg.model)

    assert isinstance(models["context_encoder"], Encoder)
    assert isinstance(models["target_encoder"], Encoder)
    assert isinstance(models["predictor"], Predictor)
    context_encoder = models["context_encoder"].to(device)
    target_encoder = models["target_encoder"].to(device)
    predictor = models["predictor"].to(device)

    # Create data pipeline via instantiate
    frame_loader = instantiate(cfg.data.frame_loader)
    frame_stacker = instantiate(cfg.data.frame_stacker)
    buffer = instantiate(cfg.data.buffer)

    # Create training components
    video_shape = tuple(cfg.model.video_shape)
    tubelet_size = tuple(cfg.model.tubelet_size)
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)

    collator = instantiate(cfg.training.collator, num_tubelets=n_tubelets)

    optimizer_factory = instantiate(cfg.training.optimizer)
    optimizer = optimizer_factory(
        list(context_encoder.parameters()) + list(predictor.parameters()),
    )

    training_logic = instantiate(
        cfg.training.training_logic,
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        optimizer=optimizer,
        ema_momentum=cfg.training.ema_momentum,
    )

    tracker = ExperimentTracker(
        enabled=cfg.get("clearml", {}).get("enabled", False),
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
        collator=collator,
        checkpoint_manager=checkpoint_manager,
        trigger_every_n_frames=cfg.training.trigger_every_n_frames,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        checkpoint_interval_seconds=cfg.training.checkpoint_interval_seconds,
        device=device,
        tracker=tracker,
    )

    # Run training
    model_dict = {
        "context_encoder": context_encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
    }
    training_loop.run(model_dict)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
