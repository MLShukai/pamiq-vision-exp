"""VAE Training Logic."""

from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, Optimizer

from exp.models.vae import VAEDecoder, VAEEncoder
from exp.trainers.base import TrainingLogic, TrainStepResult


class VAETrainingLogic(TrainingLogic):
    """Training logic for VAE baseline.

    Handles reconstruction loss (MSE) + KL divergence loss. Operates on
    the same streaming data pipeline as V-JEPA.
    """

    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        optimizer: Optimizer,
        kl_weight: float = 1e-3,
    ) -> None:
        """Initialize VAE training logic.

        Args:
            encoder: VAE encoder.
            decoder: VAE decoder.
            optimizer: Optimizer for encoder + decoder parameters.
            kl_weight: Weight for KL divergence term.
        """
        self._encoder = encoder
        self._decoder = decoder
        self._optimizer = optimizer
        self._kl_weight = kl_weight

    @override
    def train_step_from_batch(self, batch: Tensor) -> TrainStepResult:
        """Execute one VAE training step.

        Args:
            batch: Stacked videos [B, C, T, H, W].

        Returns:
            TrainStepResult with combined loss and metrics.
        """
        device = next(self._encoder.parameters()).device
        batch = batch.to(device)

        self._optimizer.zero_grad()

        # Encode
        mu, log_var = self._encoder.encode_with_logvar(batch)
        z = self._encoder.reparameterize(mu, log_var)

        # Decode
        recon = self._decoder(z)

        # Target: time-averaged frame, resized to match decoder output
        target = batch.mean(dim=2)  # [B, C, H, W]
        target_resized = F.interpolate(
            target, size=recon.shape[2:], mode="bilinear", align_corners=False
        )

        # Losses
        recon_loss = F.mse_loss(recon, target_resized)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + self._kl_weight * kl_loss

        loss.backward()
        self._optimizer.step()

        return TrainStepResult(
            loss=loss,
            metrics={
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
            },
        )


def create_vae_training_logic(
    encoder: nn.Module,
    decoder: nn.Module,
    lr: float = 1e-3,
    kl_weight: float = 1e-3,
    **kwargs: object,
) -> VAETrainingLogic:
    """Create a VAETrainingLogic with optimizer.

    Args:
        encoder: VAE encoder module.
        decoder: VAE decoder module.
        lr: Learning rate.
        kl_weight: Weight for KL divergence term.
        **kwargs: Ignored (accepts extra model dict keys gracefully).

    Returns:
        Configured VAETrainingLogic.
    """
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
    )

    return VAETrainingLogic(
        encoder=cast(VAEEncoder, encoder),
        decoder=cast(VAEDecoder, decoder),
        optimizer=optimizer,
        kl_weight=kl_weight,
    )
