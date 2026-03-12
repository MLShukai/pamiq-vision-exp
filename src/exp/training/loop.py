import logging
import time

import torch
import torch.nn as nn
from torch import Tensor

from exp.data.buffer import ReplayBuffer
from exp.data.loader import VideoFrameLoader
from exp.data.stacking import FrameStacker
from exp.trainers.vjepa.collator import VideoMultiBlockMaskCollator
from exp.trainers.vjepa.logic import VJEPATrainingLogic
from exp.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Main training loop for streaming video representation learning.

    Orchestrates the data pipeline: frames -> stacking -> buffer -> trigger -> learn.
    Saves checkpoints at regular time intervals.
    """

    def __init__(
        self,
        frame_loader: VideoFrameLoader,
        frame_stacker: FrameStacker,
        buffer: ReplayBuffer,
        training_logic: VJEPATrainingLogic,
        collator: VideoMultiBlockMaskCollator,
        checkpoint_manager: CheckpointManager,
        trigger_every_n_frames: int,
        batch_size: int,
        num_epochs: int = 1,
        checkpoint_interval_seconds: float = 300.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if trigger_every_n_frames <= 0:
            raise ValueError(
                f"trigger_every_n_frames must be positive, got {trigger_every_n_frames}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}")

        self._frame_loader = frame_loader
        self._frame_stacker = frame_stacker
        self._buffer = buffer
        self._training_logic = training_logic
        self._collator = collator
        self._checkpoint_manager = checkpoint_manager
        self._trigger_every_n_frames = trigger_every_n_frames
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._checkpoint_interval_seconds = checkpoint_interval_seconds
        self._device = device

    def run(self, models: dict[str, nn.Module]) -> None:
        """Run the full training loop.

        Args:
            models: Dictionary of model name -> module for checkpointing.
        """
        frames_since_trigger = 0
        start_time = time.monotonic()
        last_checkpoint_time = start_time
        total_frames = 0

        for frame in self._frame_loader:
            stacked = self._frame_stacker.push(frame)
            if stacked is None:
                continue

            self._buffer.add(stacked)
            frames_since_trigger += 1
            total_frames += 1

            if (
                frames_since_trigger >= self._trigger_every_n_frames
                and len(self._buffer) >= self._batch_size
            ):
                self._learn()
                frames_since_trigger = 0

            current_time = time.monotonic()
            if current_time - last_checkpoint_time >= self._checkpoint_interval_seconds:
                elapsed = current_time - start_time
                self._checkpoint_manager.save(models, elapsed)
                last_checkpoint_time = current_time
                logger.info(
                    f"Checkpoint saved at {elapsed:.1f}s, {total_frames} frames processed"
                )

        # Final checkpoint
        elapsed = time.monotonic() - start_time
        self._checkpoint_manager.save(models, elapsed)
        logger.info(f"Training complete. {total_frames} frames, {elapsed:.1f}s elapsed")

    def _learn(self) -> None:
        """Execute learning: sample batches and train for num_epochs."""
        num_steps = max(len(self._buffer) // self._batch_size, 1)

        for epoch in range(self._num_epochs):
            for step in range(num_steps):
                batch = self._buffer.sample(self._batch_size)
                # batch is [batch_size, C, T, H, W]

                # Use collator to create masks
                # The collator expects a list of video tensors
                video_list = [batch[i] for i in range(batch.shape[0])]
                videos, masks, targets = self._collator(video_list)

                # Move to device
                videos = videos.to(self._device)
                masks = masks.to(self._device)
                targets = targets.to(self._device)

                result = self._training_logic.train_step(videos, masks, targets)

                logger.debug(
                    f"Epoch {epoch}, Step {step}, Loss: {result.loss.item():.4f}"
                )
