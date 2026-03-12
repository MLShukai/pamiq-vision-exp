"""Training entry point for vision representation learning."""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from exp.data.buffer import FIFOReplayBuffer
from exp.data.loader import VideoFrameLoader
from exp.data.stacking import FrameStacker
from exp.models.components.patchfier import VideoPatchifier
from exp.models.vjepa import Encoder, Predictor, create_video_jepa
from exp.trainers.vjepa.collator import VideoMultiBlockMaskCollator
from exp.trainers.vjepa.logic import VJEPATrainingLogic
from exp.training.checkpoint import CheckpointManager
from exp.training.loop import TrainingLoop

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run training pipeline."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # Create model
    video_shape = tuple(cfg.model.video_shape)
    tubelet_size = tuple(cfg.model.tubelet_size)

    models = create_video_jepa(
        video_shape=video_shape,
        tubelet_size=tubelet_size,
        in_channels=cfg.model.in_channels,
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        predictor_hidden_dim=cfg.model.predictor_hidden_dim,
        predictor_depth=cfg.model.predictor_depth,
        predictor_num_heads=cfg.model.predictor_num_heads,
    )

    assert isinstance(models["context_encoder"], Encoder)
    assert isinstance(models["target_encoder"], Encoder)
    assert isinstance(models["predictor"], Predictor)
    context_encoder = models["context_encoder"].to(device)
    target_encoder = models["target_encoder"].to(device)
    predictor = models["predictor"].to(device)

    # Create data pipeline
    video_paths = [Path(p) for p in cfg.data.video_paths]
    frame_loader = VideoFrameLoader(
        video_paths=video_paths,
        target_fps=cfg.data.target_fps,
        target_size=tuple(cfg.data.target_size),
        mean=tuple(cfg.data.mean),
        std=tuple(cfg.data.std),
        fade_duration=cfg.data.fade_duration,
    )
    frame_stacker = FrameStacker(
        num_frames=cfg.data.num_frames,
        stride=cfg.data.stride,
    )
    buffer = FIFOReplayBuffer(max_size=cfg.data.buffer_max_size)

    # Create training components
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)
    collator = VideoMultiBlockMaskCollator(num_tubelets=n_tubelets)

    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    training_logic = VJEPATrainingLogic(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        optimizer=optimizer,
        ema_momentum=cfg.model.ema_momentum,
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
