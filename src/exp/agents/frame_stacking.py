"""Agents for vision experiments."""

from typing import override

import torch
from pamiq_core import Agent
from torch import Tensor


class FrameStackingAgent(Agent[Tensor, None]):
    """Agent that stacks N frames with a stride interval for VideoJEPA.

    This agent collects observations by stacking N frames sampled at regular
    stride intervals. For example, with num_frames=4 and frame_stride=2:
    - Step 7: collects frames [1, 3, 5, 7]
    - Step 8: collects frames [2, 4, 6, 8]
    - Step 9: collects frames [3, 5, 7, 9]
    """

    def __init__(
        self,
        num_frames: int,
        frame_stride: int,
        buffer_name: str = "video",
    ) -> None:
        """Initialize the frame stacking agent.

        Args:
            num_frames: Number of frames to stack.
            frame_stride: Stride interval between stacked frames.
            buffer_name: Name of the buffer for collecting data.
        """
        super().__init__()
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")

        self._num_frames = num_frames
        self._frame_stride = frame_stride
        self._buffer_name = buffer_name
        self._max_history = (num_frames - 1) * frame_stride + 1
        self._frame_history: list[Tensor] = []

    @override
    def on_data_collectors_attached(self) -> None:
        super().on_data_collectors_attached()
        self._collector = self.get_data_collector(self._buffer_name)

    @override
    def step(self, observation: Tensor) -> None:
        self._frame_history.append(observation)

        if len(self._frame_history) > self._max_history:
            self._frame_history.pop(0)

        if len(self._frame_history) < self._max_history:
            return

        stacked_frames = self._sample_frames()
        stacked_tensor = torch.stack(stacked_frames, dim=0)
        self._collector.collect(stacked_tensor)

        return

    def _sample_frames(self) -> list[Tensor]:
        """Sample frames at stride intervals from history."""
        frames = []
        for i in range(self._num_frames):
            idx = -(self._max_history - i * self._frame_stride)
            frames.append(self._frame_history[idx])
        return frames
