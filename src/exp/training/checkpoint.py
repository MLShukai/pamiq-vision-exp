import time
from pathlib import Path

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages saving and loading model checkpoints.

    Checkpoints are saved in the directory structure:
    log_dir/experiment_name/<start_timestamp>/checkpoints/<elapsed_seconds>.ckpt/
    """

    def __init__(self, log_dir: Path, experiment_name: str) -> None:
        self._start_time = time.strftime("%Y%m%d_%H%M%S")
        self._checkpoint_dir = (
            log_dir / experiment_name / self._start_time / "checkpoints"
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        """Base directory for all checkpoints."""
        return self._checkpoint_dir

    def save(
        self,
        models: dict[str, nn.Module],
        elapsed_seconds: float,
        extra: dict[str, object] | None = None,
    ) -> Path:
        """Save model checkpoints.

        Args:
            models: Dictionary mapping model names to modules.
            elapsed_seconds: Elapsed training time in seconds.
            extra: Optional extra data to save.

        Returns:
            Path to the checkpoint directory.
        """
        ckpt_name = f"{int(elapsed_seconds)}.ckpt"
        ckpt_path = self._checkpoint_dir / ckpt_name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        for name, model in models.items():
            torch.save(model.state_dict(), ckpt_path / f"{name}.pt")

        if extra is not None:
            torch.save(extra, ckpt_path / "extra.pt")

        return ckpt_path

    @staticmethod
    def load(
        checkpoint_path: Path,
        models: dict[str, nn.Module],
    ) -> None:
        """Load model parameters from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
            models: Dictionary mapping model names to modules to load into.
        """
        for name, model in models.items():
            state_path = checkpoint_path / f"{name}.pt"
            if not state_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {state_path}")
            state_dict = torch.load(state_path, weights_only=True)
            model.load_state_dict(state_dict)

    @staticmethod
    def load_encoder(
        checkpoint_path: Path,
        encoder_key: str,
        encoder: nn.Module,
    ) -> nn.Module:
        """Load a single encoder from checkpoint for evaluation.

        Args:
            checkpoint_path: Path to checkpoint directory.
            encoder_key: Key name of the encoder in the checkpoint.
            encoder: Encoder module to load parameters into.

        Returns:
            The encoder with loaded parameters in eval mode.
        """
        state_path = checkpoint_path / f"{encoder_key}.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {state_path}")
        state_dict = torch.load(state_path, weights_only=True)
        encoder.load_state_dict(state_dict)
        encoder.eval()
        return encoder
