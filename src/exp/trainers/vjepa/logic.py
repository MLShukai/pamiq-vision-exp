"""V-JEPA Training Logic."""

from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer

from exp.models.components.patchfier import VideoPatchifier
from exp.models.vjepa import Encoder, Predictor
from exp.trainers.base import TrainingLogic, TrainStepResult
from exp.trainers.vjepa.collator import VideoMultiBlockMaskCollator


@dataclass
class VJEPALossResult:
    """Result of V-JEPA loss computation."""

    loss: Tensor
    """Averaged L1 loss over predicted tubelets."""

    loss_per_data: Tensor
    """L1 loss per sample in batch."""

    latent_context: Tensor
    """Context encoder representations."""

    latent_target: Tensor
    """Target encoder representations (normalized)."""

    latent_predictor: Tensor
    """Predictor output representations."""


class VJEPATrainingLogic(TrainingLogic):
    """Core training logic for V-JEPA.

    This class encapsulates the V-JEPA training algorithm including:
    1. Loss computation using L1 distance between predicted and target representations
    2. Exponential moving average (EMA) update of the target encoder
    3. Single training step execution

    The training uses a masked prediction task where the model learns to predict
    representations of target tubelets from context tubelets.
    """

    def __init__(
        self,
        context_encoder: Encoder,
        target_encoder: Encoder,
        predictor: Predictor,
        optimizer: Optimizer,
        collator: VideoMultiBlockMaskCollator,
        ema_momentum: float = 0.998,
    ) -> None:
        """Initialize the V-JEPA training logic.

        Args:
            context_encoder: Context encoder that processes masked inputs.
            target_encoder: Target encoder for generating prediction targets.
            predictor: Predictor that predicts target representations.
            optimizer: Optimizer for context encoder and predictor parameters.
            collator: Mask collator for generating V-JEPA masks.
            ema_momentum: Momentum coefficient for EMA update of target encoder.
        """
        self._context_encoder = context_encoder
        self._target_encoder = target_encoder
        self._predictor = predictor
        self._optimizer = optimizer
        self._collator = collator
        self._ema_momentum = ema_momentum

    def compute_loss(
        self,
        videos: Tensor,
        masks_for_context_encoder: Tensor,
        targets_for_predictor: Tensor,
    ) -> VJEPALossResult:
        """Compute V-JEPA loss between predicted and target representations.

        Args:
            videos: Input videos [batch, channels, time, height, width].
            masks_for_context_encoder: Boolean masks for context encoder
                [batch, n_tubelets], True = masked.
            targets_for_predictor: Boolean masks indicating which tubelets
                to predict [batch, n_tubelets], True = predict.

        Returns:
            VJEPALossResult containing loss values and intermediate representations.
        """
        # Target encoder (no gradient)
        with torch.no_grad():
            latent_target = self._target_encoder.encode_tubelets(videos)
            # Normalize over feature dimension (V-JEPA uses LayerNorm)
            latent_target = F.layer_norm(
                latent_target,
                (latent_target.size(-1),),
            )

        # Context encoder with masking
        latent_context = self._context_encoder.encode_tubelets(
            videos, masks_for_context_encoder
        )

        # Predictor
        latent_predictor = self._predictor(latent_context, targets_for_predictor)

        # L1 loss (V-JEPA paper Equation 2)
        losses = F.l1_loss(latent_predictor, latent_target, reduction="none").mean(-1)
        # Shape: [batch, n_tubelets]

        # Ignore tubelets that are not selected for prediction
        losses = torch.masked_fill(losses, ~targets_for_predictor, 0.0)
        loss_per_data = losses.sum(-1) / targets_for_predictor.sum(-1)
        loss = losses.sum() / targets_for_predictor.sum()

        return VJEPALossResult(
            loss=loss,
            loss_per_data=loss_per_data,
            latent_context=latent_context,
            latent_target=latent_target,
            latent_predictor=latent_predictor,
        )

    def update_target_encoder(self) -> None:
        """Update target encoder parameters using EMA from context encoder."""
        with torch.no_grad():
            for target_param, context_param in zip(
                self._target_encoder.parameters(),
                self._context_encoder.parameters(),
                strict=True,
            ):
                target_param.data.mul_(self._ema_momentum).add_(
                    (1.0 - self._ema_momentum) * context_param.detach().data
                )

    def train_step(
        self,
        videos: Tensor,
        masks_for_context_encoder: Tensor,
        targets_for_predictor: Tensor,
    ) -> VJEPALossResult:
        """Execute a single training step with pre-computed masks.

        Args:
            videos: Input videos [batch, channels, time, height, width].
            masks_for_context_encoder: Boolean masks for context encoder.
            targets_for_predictor: Boolean masks for predictor target.

        Returns:
            VJEPALossResult containing loss values and intermediate representations.
        """
        self._optimizer.zero_grad()

        result = self.compute_loss(
            videos,
            masks_for_context_encoder,
            targets_for_predictor,
        )

        result.loss.backward()
        self._optimizer.step()

        # Update target encoder with EMA from context encoder
        self.update_target_encoder()

        return result

    @override
    def train_step_from_batch(self, batch: Tensor) -> TrainStepResult:
        """Execute a training step from raw video batch.

        Handles collation (mask generation) internally.

        Args:
            batch: Stacked videos [batch_size, C, T, H, W].

        Returns:
            TrainStepResult with loss and metrics.
        """
        videos, masks, targets = self._collator(list(batch.unbind(0)))

        device = next(self._context_encoder.parameters()).device
        videos, masks, targets = (
            videos.to(device),
            masks.to(device),
            targets.to(device),
        )

        result = self.train_step(videos, masks, targets)

        return TrainStepResult(
            loss=result.loss,
            metrics={
                "target_std": result.latent_target.std(0).mean().item(),
                "context_std": result.latent_context.std(0).mean().item(),
            },
        )


def create_vjepa_training_logic(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    video_shape: tuple[int, int, int] = (16, 224, 224),
    tubelet_size: tuple[int, int, int] = (2, 16, 16),
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    ema_momentum: float = 0.996,
    **kwargs: object,
) -> VJEPATrainingLogic:
    """Create a VJEPATrainingLogic with optimizer and collator.

    Args:
        context_encoder: Context encoder module.
        target_encoder: Target encoder module (EMA copy).
        predictor: Predictor module.
        video_shape: Video shape (T, H, W) for computing tubelets.
        tubelet_size: Tubelet size (T, H, W).
        lr: Learning rate.
        weight_decay: Weight decay.
        ema_momentum: EMA momentum for target encoder.

    Returns:
        Configured VJEPATrainingLogic.
    """
    assert isinstance(context_encoder, Encoder)
    assert isinstance(target_encoder, Encoder)
    assert isinstance(predictor, Predictor)

    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)
    collator = VideoMultiBlockMaskCollator(num_tubelets=n_tubelets)

    optimizer = AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    return VJEPATrainingLogic(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        optimizer=optimizer,
        collator=collator,
        ema_momentum=ema_momentum,
    )
