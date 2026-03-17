import logging
from pathlib import Path
from typing import override

import pytest
import torch
from torch import Tensor

from exp.trainers.base import TrainingLogic, TrainStepResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

mark_slow = pytest.mark.slow


CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device("cuda:0")


def get_available_devices() -> list[torch.device]:
    devices = [CPU_DEVICE]
    if torch.cuda.is_available():
        devices.append(CUDA_DEVICE)
    return devices


logger.info("Available devices: " + ", ".join(map(str, get_available_devices())))

parametrize_device = pytest.mark.parametrize("device", get_available_devices())


class TrainingLogicImpl(TrainingLogic):
    """Concrete TrainingLogic for testing."""

    def __init__(self) -> None:
        self.call_count = 0

    @override
    def train_step_from_batch(self, batch: Tensor) -> TrainStepResult:
        self.call_count += 1
        return TrainStepResult(loss=torch.tensor(0.5), metrics={})


def create_test_video(
    path: Path,
    n_frames: int,
    height: int = 240,
    width: int = 320,
    fps: float = 30.0,
    pixel_value: int | None = None,
) -> None:
    """テスト用の動画ファイルを生成する。"""
    from fractions import Fraction

    import av
    import numpy as np

    container = av.open(str(path), "w")
    stream = container.add_stream(
        "rawvideo", rate=Fraction(fps).limit_denominator(1000)
    )
    video_stream: av.VideoStream = stream  # type: ignore[assignment]
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = "rgb24"

    for _ in range(n_frames):
        if pixel_value is not None:
            data = np.full((height, width, 3), pixel_value, dtype=np.uint8)
        else:
            data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(data, format="rgb24")
        container.mux(video_stream.encode(frame))

    for packet in video_stream.encode():
        container.mux(packet)
    container.close()
