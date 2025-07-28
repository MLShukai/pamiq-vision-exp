"""Tests for MetricsLogger base class."""

from typing import override

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from exp.validation.metrics_loggers.base import MetricsLogger


class SimpleMetricsLogger(MetricsLogger):
    """Simple implementation for testing."""

    def __init__(self):
        super().__init__(metrics_type="test")
        self.dataloader_called = False
        self.batch_count = 0

    @override
    def create_dataloader(self) -> DataLoader:
        """Create a simple dataloader."""
        self.dataloader_called = True
        return DataLoader(self.dataset, batch_size=2)

    @override
    def batch_step(self, batch, index):
        """Count batches."""
        self.batch_count += 1


class TestMetricsLogger:
    """Test MetricsLogger base functionality."""

    def test_dataset_attachment(self):
        """Test dataset property and attachment."""
        logger = SimpleMetricsLogger()

        # Should raise error before attachment
        with pytest.raises(RuntimeError, match="Dataset not attached"):
            _ = logger.dataset

        # Should work after attachment
        dataset = TensorDataset(torch.randn(10, 3))
        logger.attach_dataset(dataset)
        assert logger.dataset is dataset

    def test_models_attachment(self):
        """Test models property and attachment."""
        logger = SimpleMetricsLogger()

        # Should raise error before attachment
        with pytest.raises(RuntimeError, match="Models not attached"):
            _ = logger.models

        # Should work after attachment
        models = {"model": nn.Linear(10, 5)}
        logger.attach_models(models)
        assert logger.models is models

    def test_exp_cfg_attachment(self):
        """Test experiment config attachment."""
        logger = SimpleMetricsLogger()

        # Should raise error before attachment
        with pytest.raises(RuntimeError, match="Experiemnt config not attached"):
            _ = logger.exp_cfg

        # Should work after attachment
        cfg = DictConfig({"test": "value"})
        logger.attach_exp_cfg(cfg)
        assert logger.exp_cfg is cfg

    def test_run_without_aim(self):
        """Test run fails without Aim setup."""
        logger = SimpleMetricsLogger()
        dataset = TensorDataset(torch.randn(4, 3))
        logger.attach_dataset(dataset)

        with pytest.raises(ValueError, match="Aim run not initialized"):
            logger.run()
