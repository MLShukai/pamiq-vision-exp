"""V-JEPA Training Logic - PAMIQ Core independent implementation.

This module provides the core training logic for Video Joint Embedding
Predictive Architecture (V-JEPA) without any dependency on PAMIQ Core.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from exp.models.vjepa import Encoder, Predictor


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


class VJEPATrainingLogic:
    """Core training logic for V-JEPA, independent of PAMIQ Core.

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
        ema_momentum: float = 0.998,
    ) -> None:
        """Initialize the V-JEPA training logic.

        Args:
            context_encoder: Context encoder that processes masked inputs.
            target_encoder: Target encoder for generating prediction targets.
            predictor: Predictor that predicts target representations.
            optimizer: Optimizer for context encoder and predictor parameters.
            ema_momentum: Momentum coefficient for EMA update of target encoder.
                Higher values mean slower updates. Default 0.998 based on V-JEPA paper.
        """
        self._context_encoder = context_encoder
        self._target_encoder = target_encoder
        self._predictor = predictor
        self._optimizer = optimizer
        self._ema_momentum = ema_momentum

    def compute_loss(
        self,
        videos: Tensor,
        masks_for_context_encoder: Tensor,
        targets_for_predictor: Tensor,
    ) -> VJEPALossResult:
        """Compute V-JEPA loss between predicted and target representations.

        Following V-JEPA paper (Equation 2), the loss is the average L1 distance
        between the predictor output and the normalized target encoder output
        for masked tubelets.

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
            latent_target = self._target_encoder(videos)
            # Normalize over feature dimension (V-JEPA uses LayerNorm)
            latent_target = F.layer_norm(
                latent_target,
                (latent_target.size(-1),),
            )

        # Context encoder with masking
        latent_context = self._context_encoder(videos, masks_for_context_encoder)

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
        """Execute a single training step.

        This method performs:
        1. Forward pass through context encoder, predictor, and target encoder
        2. Loss computation using L1 distance
        3. Backward pass and optimizer step
        4. EMA update of target encoder

        Args:
            videos: Input videos [batch, channels, time, height, width].
            masks_for_context_encoder: Boolean masks for context encoder
                [batch, n_tubelets], True = masked.
            targets_for_predictor: Boolean masks indicating which tubelets
                to predict [batch, n_tubelets], True = predict.

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
