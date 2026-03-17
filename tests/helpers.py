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
