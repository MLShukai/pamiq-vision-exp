from typing import override

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from exp.validation.metrics_loggers.base import MetricsLogger


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __len__(self):
        return 10

    @override
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), idx


class DummyMetricsLogger(MetricsLogger):
    """Dummy implementation of MetricsLogger for testing."""

    @override
    def run(self) -> None:
        """Dummy implementation of run method."""
        # Access dataset and models to ensure they work
        _ = self.dataset
        _ = self.models


class TestMetricsLogger:
    """Test the MetricsLogger base class."""

    @pytest.fixture
    def dummy_dataset(self):
        """Create a dummy dataset."""
        return DummyDataset()

    @pytest.fixture
    def dummy_models(self):
        """Create dummy models."""
        return {
            "encoder": nn.Linear(10, 5),
            "decoder": nn.Linear(5, 10),
        }

    @pytest.fixture
    def logger(self):
        """Create a dummy metrics logger."""
        return DummyMetricsLogger()

    def test_dataset_property_error_when_not_attached(self, logger):
        """Test that accessing dataset property raises error when not
        attached."""
        with pytest.raises(
            RuntimeError, match="Dataset not attached. Call attach_dataset\\(\\) first."
        ):
            _ = logger.dataset

    def test_models_property_error_when_not_attached(self, logger):
        """Test that accessing models property raises error when not
        attached."""
        with pytest.raises(
            RuntimeError, match="Models not attached. Call attach_models\\(\\) first."
        ):
            _ = logger.models
