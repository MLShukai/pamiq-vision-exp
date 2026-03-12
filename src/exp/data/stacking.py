from collections import deque

import torch
from torch import Tensor


class FrameStacker:
    """Stacks consecutive frames with stride for video model input."""

    def __init__(self, num_frames: int, stride: int = 1) -> None:
        """Initialize frame stacker.

        Args:
            num_frames: Number of frames to stack.
            stride: Stride interval between frames.

        Raises:
            ValueError: If num_frames or stride is not positive.
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self._num_frames = num_frames
        self._stride = stride
        self._history_size = (num_frames - 1) * stride + 1
        self._history: deque[Tensor] = deque(maxlen=self._history_size)

    def push(self, frame: Tensor) -> Tensor | None:
        """Add a frame and return stacked frames if enough accumulated.

        Args:
            frame: Single frame tensor [C, H, W].

        Returns:
            Stacked frames [C, T, H, W] if enough frames accumulated, else None.
        """
        self._history.append(frame)

        if len(self._history) < self._history_size:
            return None

        frames = [self._history[i * self._stride] for i in range(self._num_frames)]
        # Each frame is [C, H, W]. stack on dim=1 inserts T at position 1 -> [C, T, H, W]
        return torch.stack(frames, dim=1)

    def reset(self) -> None:
        """Clear frame history."""
        self._history.clear()
