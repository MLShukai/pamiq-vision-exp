import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """Results from reconstruction evaluation."""

    mae: float
    mse: float


class ReconstructionEvaluator:
    """Evaluates representation quality via input reconstruction."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._encoder = encoder
        self._decoder = decoder
        self._device = device

    @torch.no_grad()
    def encode_dataset(
        self,
        videos: Tensor,
        batch_size: int = 32,
    ) -> Tensor:
        """Encode all videos to feature representations.

        Args:
            videos: Video tensors [N, C, T, H, W]
            batch_size: Batch size for encoding

        Returns:
            Encoded features [N, n_tubelets, embed_dim]
        """
        self._encoder.eval()
        features_list = []

        for i in range(0, len(videos), batch_size):
            batch = videos[i : i + batch_size].to(self._device)
            features = self._encoder(batch)
            features_list.append(features.cpu())

        return torch.cat(features_list, dim=0)

    def train_decoder(
        self,
        features: Tensor,
        targets: Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train decoder on frozen features.

        Args:
            features: Encoded features [N, n_tubelets, embed_dim]
            targets: Original videos [N, C, T, H, W]
            num_epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate

        Returns:
            List of loss values per epoch
        """
        self._decoder.to(self._device)
        self._decoder.train()
        optimizer = torch.optim.Adam(self._decoder.parameters(), lr=lr)

        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            count = 0
            for feat_batch, target_batch in dataloader:
                feat_batch = feat_batch.to(self._device)
                target_batch = target_batch.to(self._device)

                optimizer.zero_grad()
                reconstructed = self._decoder(feat_batch)
                loss = F.mse_loss(reconstructed, target_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(feat_batch)
                count += len(feat_batch)

            avg_loss = epoch_loss / count
            losses.append(avg_loss)
            logger.debug(f"Decoder epoch {epoch}: loss={avg_loss:.6f}")

        return losses

    @torch.no_grad()
    def evaluate(
        self,
        features: Tensor,
        targets: Tensor,
        batch_size: int = 32,
    ) -> ReconstructionResult:
        """Evaluate reconstruction quality.

        Args:
            features: Encoded features [N, n_tubelets, embed_dim]
            targets: Original videos [N, C, T, H, W]
            batch_size: Batch size for evaluation

        Returns:
            ReconstructionResult with MAE and MSE metrics.
        """
        self._decoder.eval()
        self._decoder.to(self._device)

        total_mae = 0.0
        total_mse = 0.0
        count = 0

        for i in range(0, len(features), batch_size):
            feat_batch = features[i : i + batch_size].to(self._device)
            target_batch = targets[i : i + batch_size].to(self._device)

            reconstructed = self._decoder(feat_batch)
            total_mae += F.l1_loss(reconstructed, target_batch, reduction="sum").item()
            total_mse += F.mse_loss(reconstructed, target_batch, reduction="sum").item()
            count += target_batch.numel()

        return ReconstructionResult(
            mae=total_mae / count,
            mse=total_mse / count,
        )
