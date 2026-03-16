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

    def get_data(self) -> Tensor:
        """Return the valid data currently stored in the buffer.

        Returns:
            Tensor containing all valid items in the buffer.

        Raises:
            RuntimeError: If the buffer is empty.
        """
        if self._current_size == 0 or self._storage is None:
            raise RuntimeError("Cannot get data from empty buffer")
        return self._storage[: self._current_size]

    def __len__(self) -> int:
        return self._current_size

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
