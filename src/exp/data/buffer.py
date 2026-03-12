from pathlib import Path

import torch
from torch import Tensor


class FIFOReplayBuffer:
    """FIFO replay buffer with pre-allocated tensor storage."""

    def __init__(self, max_size: int) -> None:
        """Initialize FIFO replay buffer.

        Args:
            max_size: Maximum number of items to store.

        Raises:
            ValueError: If max_size is not positive.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = max_size
        self._storage: Tensor | None = None
        self._write_index = 0
        self._current_size = 0

    def add(self, item: Tensor) -> None:
        """Add an item to the buffer.

        On the first call, storage is pre-allocated based on the item shape.

        Args:
            item: Tensor to add.
        """
        if self._storage is None:
            self._storage = torch.empty(self._max_size, *item.shape, dtype=item.dtype)

        idx = self._write_index % self._max_size
        self._storage[idx] = item
        self._write_index += 1
        self._current_size = min(self._current_size + 1, self._max_size)

    def sample(self, batch_size: int) -> Tensor:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of items to sample.

        Returns:
            Batch tensor of sampled items.

        Raises:
            RuntimeError: If the buffer is empty.
            ValueError: If batch_size exceeds current buffer size.
        """
        if self._current_size == 0 or self._storage is None:
            raise RuntimeError("Cannot sample from empty buffer")
        if batch_size > self._current_size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self._current_size})"
            )

        indices = torch.randint(0, self._current_size, (batch_size,))
        return self._storage[indices]

    def __len__(self) -> int:
        return self._current_size

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached max capacity."""
        return self._current_size >= self._max_size

    def save(self, path: Path) -> None:
        """Save buffer state to disk.

        Args:
            path: Directory to save buffer state into.
        """
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "storage": self._storage,
            "write_index": self._write_index,
            "current_size": self._current_size,
            "max_size": self._max_size,
        }
        torch.save(state, path / "buffer.pt")

    def load(self, path: Path) -> None:
        """Load buffer state from disk.

        Args:
            path: Directory containing saved buffer state.
        """
        state = torch.load(path / "buffer.pt", weights_only=False)
        self._storage = state["storage"]
        self._write_index = state["write_index"]
        self._current_size = state["current_size"]
        self._max_size = state["max_size"]
