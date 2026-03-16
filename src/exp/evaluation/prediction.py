import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Results from future prediction evaluation."""

    horizon_errors: dict[int, float]
    """MAE for each prediction horizon."""
    pointwise_horizon_errors: dict[int, Tensor]
    """Per-sequence MAE for each horizon.

    Values have shape [num_seqs].
    """


class PredictionEvaluator:
    """Evaluates representation quality via future prediction in feature
    space."""

    def __init__(
        self,
        predictor: nn.Module,
        horizons: list[int] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._predictor = predictor
        self._horizons = horizons or [1, 2, 4, 8]
        self._device = device

    def create_sequences(
        self,
        features: Tensor,
        seq_len: int,
    ) -> tuple[Tensor, Tensor]:
        """Create input-target sequence pairs from feature time series.

        Args:
            features: Feature time series [N, feat_dim] (flattened per-frame features)
            seq_len: Length of each input sequence

        Returns:
            Tuple of (inputs [num_seqs, seq_len, feat_dim], targets [num_seqs, max_horizon, feat_dim])
        """
        max_horizon = max(self._horizons)
        num_seqs = len(features) - seq_len - max_horizon + 1

        if num_seqs <= 0:
            raise ValueError(
                f"Not enough features ({len(features)}) for seq_len={seq_len} "
                f"and max_horizon={max_horizon}"
            )

        inputs = torch.stack([features[i : i + seq_len] for i in range(num_seqs)])
        targets = torch.stack(
            [features[i + seq_len : i + seq_len + max_horizon] for i in range(num_seqs)]
        )

        return inputs, targets

    def train_predictor(
        self,
        features: Tensor,
        seq_len: int = 32,
        num_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train predictor on feature sequences.

        Args:
            features: Feature time series [N, feat_dim]
            seq_len: Input sequence length
            num_epochs: Training epochs
            batch_size: Training batch size
            lr: Learning rate

        Returns:
            List of loss values per epoch
        """
        inputs, targets = self.create_sequences(features, seq_len)

        self._predictor.to(self._device)
        self._predictor.train()
        optimizer = torch.optim.Adam(self._predictor.parameters(), lr=lr)

        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            count = 0
            for inp_batch, tgt_batch in dataloader:
                inp_batch = inp_batch.to(self._device)
                tgt_batch = tgt_batch.to(self._device)

                optimizer.zero_grad()
                # Predict max_horizon steps ahead
                output, _ = self._predictor(inp_batch)
                # Use last max_horizon outputs as predictions
                max_horizon = max(self._horizons)
                pred = output[:, -max_horizon:]
                loss = F.l1_loss(pred, tgt_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(inp_batch)
                count += len(inp_batch)

            losses.append(epoch_loss / count)

        return losses

    @torch.no_grad()
    def evaluate(
        self,
        features: Tensor,
        seq_len: int = 32,
        batch_size: int = 32,
    ) -> PredictionResult:
        """Evaluate prediction quality at each horizon.

        Args:
            features: Feature time series [N, feat_dim]
            seq_len: Input sequence length
            batch_size: Evaluation batch size

        Returns:
            PredictionResult with per-horizon MAE.
        """
        inputs, targets = self.create_sequences(features, seq_len)

        self._predictor.eval()
        self._predictor.to(self._device)

        max_horizon = max(self._horizons)

        all_preds = []
        for i in range(0, len(inputs), batch_size):
            inp_batch = inputs[i : i + batch_size].to(self._device)
            output, _ = self._predictor(inp_batch)
            pred = output[:, -max_horizon:]
            all_preds.append(pred.cpu())

        all_preds_tensor = torch.cat(all_preds, dim=0)

        horizon_errors: dict[int, float] = {}
        pointwise_horizon_errors: dict[int, Tensor] = {}
        for h in self._horizons:
            if h > max_horizon:
                continue
            # h-step ahead prediction is at index h-1
            pred_h = all_preds_tensor[:, h - 1]
            target_h = targets[:, h - 1]
            mae_per_seq = (pred_h - target_h).abs().mean(dim=-1)
            pointwise_horizon_errors[h] = mae_per_seq
            horizon_errors[h] = mae_per_seq.mean().item()

        return PredictionResult(
            horizon_errors=horizon_errors,
            pointwise_horizon_errors=pointwise_horizon_errors,
        )
