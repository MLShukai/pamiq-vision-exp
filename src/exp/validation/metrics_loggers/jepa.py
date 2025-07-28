from typing import override

import hydra
from pamiq_core.torch import get_device
from torch import Tensor
from torch.utils.data import DataLoader

from exp.models import ModelName
from exp.models.jepa import Encoder, Predictor
from exp.trainers.jepa import JEPATrainer

from .base import MetricsLogger


class JEPAMetrics(MetricsLogger):
    """Computes and logs per-sample JEPA losses."""

    def __init__(self, batch_size: int, log_prefix: str = "jepa") -> None:
        super().__init__("jepa")
        self.batch_size = batch_size
        self.log_prefix = log_prefix

    @override
    def setup(self) -> None:
        """Initialize models and device."""
        # Get required models from registry
        context_encoder = self.models[ModelName.JEPA_CONTEXT_ENCODER]
        target_encoder = self.models[ModelName.JEPA_TARGET_ENCODER]
        predictor = self.models[ModelName.JEPA_PREDICTOR]
        # Validate model types
        if not (
            isinstance(context_encoder, Encoder)
            and isinstance(target_encoder, Encoder)
            and isinstance(predictor, Predictor)
        ):
            raise ValueError(
                f"Invalid model types. Expected (Encoder, Encoder, Predictor), "
                f"got ({type(context_encoder).__name__}, {type(target_encoder).__name__}, "
                f"{type(predictor).__name__})"
            )

        # Determine computation device from context encoder
        self.device = get_device(context_encoder)
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor

        # Store data step.
        self.data_index = 0

    @override
    def create_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        """Create dataloader with JEPA collate function."""
        # Setup DataLoader with JEPA-specific collate function
        collate_fn = hydra.utils.instantiate(self.exp_cfg.trainers.jepa.collate_fn)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,  # No shuffling for consistent metrics
            drop_last=False,  # Process all samples
        )

    @override
    def batch_step(self, batch: tuple[Tensor, Tensor, Tensor], index: int) -> None:
        """Compute and log JEPA loss for each sample in batch."""
        # Move all tensors to device
        batch = tuple(map(lambda x: x.to(self.device), batch))  # pyright: ignore[reportAssignmentType, ]
        # Compute JEPA loss for this batch
        loss_dict = JEPATrainer.compute_loss(
            self.target_encoder,
            self.context_encoder,
            self.predictor,
            batch,
        )

        # Log individual sample losses
        for loss in loss_dict["loss_per_data"]:
            self.aim_run.track(
                loss.cpu().item(),
                "loss",
                step=self.data_index,
                context=self.default_aim_context,
            )
            self.data_index += 1  # Increment index for each sample
