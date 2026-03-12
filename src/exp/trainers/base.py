"""Common training interface for all representation learning methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


@dataclass
class TrainStepResult:
    """Common result from a training step."""

    loss: Tensor
    """Scalar loss value."""

    metrics: dict[str, float]
    """Additional metrics to log (e.g., std values, KL loss)."""


class TrainingLogic(ABC):
    """Abstract base class for method-agnostic training logic.

    Each representation learning method subclasses this, handling its
    own collation/masking internally.
    """

    @abstractmethod
    def train_step_from_batch(self, batch: Tensor) -> TrainStepResult:
        """Execute one training step from a raw batch of stacked videos.

        Args:
            batch: Stacked videos [batch_size, C, T, H, W].

        Returns:
            TrainStepResult with loss and optional metrics.
        """
        ...
