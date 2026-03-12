"""VAE baseline evaluator for reconstruction comparison."""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from exp.models.vae import SimpleVAE

logger = logging.getLogger(__name__)


@dataclass
class VAEBaselineResult:
    """Results from VAE baseline evaluation."""

    recon_mae: float
    recon_mse: float
    final_train_loss: float


class VAEBaselineEvaluator:
    """Trains a simple VAE and evaluates reconstruction quality.

    Provides a learned baseline for comparing against V-JEPA
    representations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._vae = SimpleVAE(in_channels=in_channels, latent_dim=latent_dim)
        self._device = device

    def train_vae(
        self,
        videos: Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        kl_weight: float = 1e-3,
    ) -> list[float]:
        """Train VAE on video frames.

        Args:
            videos: Video tensors [N, C, T, H, W].
            num_epochs: Training epochs.
            batch_size: Batch size.
            lr: Learning rate.
            kl_weight: Weight for KL divergence term.

        Returns:
            List of loss values per epoch.
        """
        # Average over time to get frames
        frames = videos.mean(dim=2)  # [N, C, H, W]

        # Resize to 32x32 for the VAE
        frames_resized = F.interpolate(
            frames, size=(32, 32), mode="bilinear", align_corners=False
        )

        self._vae.to(self._device)
        self._vae.train()
        optimizer = torch.optim.Adam(self._vae.parameters(), lr=lr)

        dataset = TensorDataset(frames_resized)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            count = 0
            for (batch,) in dataloader:
                batch = batch.to(self._device)
                optimizer.zero_grad()

                recon, mu, log_var = self._vae(batch)
                recon_loss = F.mse_loss(recon, batch, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_weight * kl_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                count += len(batch)

            avg_loss = epoch_loss / count
            losses.append(avg_loss)
            logger.debug(f"VAE epoch {epoch}: loss={avg_loss:.6f}")

        return losses

    @torch.no_grad()
    def evaluate(self, videos: Tensor, batch_size: int = 32) -> VAEBaselineResult:
        """Evaluate VAE reconstruction quality.

        Args:
            videos: Video tensors [N, C, T, H, W].
            batch_size: Batch size for evaluation.

        Returns:
            VAEBaselineResult with reconstruction metrics.
        """
        frames = videos.mean(dim=2)
        frames_resized = F.interpolate(
            frames, size=(32, 32), mode="bilinear", align_corners=False
        )

        self._vae.eval()
        self._vae.to(self._device)

        total_mae = 0.0
        total_mse = 0.0
        count = 0

        for i in range(0, len(frames_resized), batch_size):
            batch = frames_resized[i : i + batch_size].to(self._device)
            recon, _, _ = self._vae(batch)
            total_mae += F.l1_loss(recon, batch, reduction="sum").item()
            total_mse += F.mse_loss(recon, batch, reduction="sum").item()
            count += batch.numel()

        return VAEBaselineResult(
            recon_mae=total_mae / count,
            recon_mse=total_mse / count,
            final_train_loss=0.0,
        )
