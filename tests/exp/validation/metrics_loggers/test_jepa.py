"""Tests for JEPAMetrics."""

import pytest
import torch
from omegaconf import DictConfig
from pytest_mock import MockerFixture
from torch.utils.data import TensorDataset

from exp.models import ModelName
from exp.models.jepa import Encoder, Predictor
from exp.validation.metrics_loggers.jepa import JEPAMetrics


class TestJEPAMetrics:
    """Test JEPAMetrics functionality."""

    @pytest.fixture
    def mock_models(self, mocker: MockerFixture):
        """Create mock JEPA models."""
        context_encoder = mocker.Mock(spec=Encoder)
        target_encoder = mocker.Mock(spec=Encoder)
        predictor = mocker.Mock(spec=Predictor)

        return {
            ModelName.JEPA_CONTEXT_ENCODER: context_encoder,
            ModelName.JEPA_TARGET_ENCODER: target_encoder,
            ModelName.JEPA_PREDICTOR: predictor,
        }

    @pytest.fixture
    def mock_config(self, mocker: MockerFixture):
        """Create mock experiment config."""
        cfg = DictConfig(
            {"trainers": {"jepa": {"collate_fn": {"_target_": "some.collate.fn"}}}}
        )
        return cfg

    def test_basic_run(self, mocker: MockerFixture, mock_models, mock_config):
        """Test basic metrics computation flow."""
        # Setup mocks
        mock_aim_run = mocker.Mock()
        mocker.patch(
            "exp.validation.metrics_loggers.base.get_global_run",
            return_value=mock_aim_run,
        )
        mocker.patch(
            "exp.validation.metrics_loggers.jepa.get_device",
            return_value=torch.device("cpu"),
        )

        # Mock collate function - should return tuple of 3 tensors
        def mock_collate(batch):
            # Extract tensors from TensorDataset tuples and stack
            tensors = [item[0] for item in batch]
            stacked = torch.stack(tensors)
            return stacked, stacked.clone(), stacked.clone()

        mocker.patch("hydra.utils.instantiate", return_value=mock_collate)

        # Mock loss computation
        mocker.patch(
            "exp.validation.metrics_loggers.jepa.JEPATrainer.compute_loss",
            return_value={
                "loss": torch.tensor(0.5),
                "loss_per_data": [torch.tensor(0.4), torch.tensor(0.6)],
            },
        )

        # Create and run metrics
        metrics = JEPAMetrics(batch_size=2)
        dataset = TensorDataset(torch.randn(4, 3, 32, 32))

        metrics.attach_dataset(dataset)
        metrics.attach_models(mock_models)
        metrics.attach_exp_cfg(mock_config)

        metrics.run()

        # Verify aim tracking was called
        assert mock_aim_run.track.called
        assert mock_aim_run.track.call_count == 4  # 4 samples in dataset

    def test_invalid_model_types(self, mocker: MockerFixture, mock_config):
        """Test error on invalid model types."""
        # Setup minimal mocks
        mocker.patch(
            "exp.validation.metrics_loggers.base.get_global_run",
            return_value=mocker.Mock(),
        )

        # Create metrics with wrong model types
        metrics = JEPAMetrics(batch_size=2)
        metrics.attach_dataset(TensorDataset(torch.randn(2, 3, 32, 32)))
        metrics.attach_exp_cfg(mock_config)
        metrics.attach_models(
            {
                ModelName.JEPA_CONTEXT_ENCODER: mocker.Mock(),  # Not an Encoder
                ModelName.JEPA_TARGET_ENCODER: mocker.Mock(spec=Encoder),
                ModelName.JEPA_PREDICTOR: mocker.Mock(spec=Predictor),
            }
        )

        with pytest.raises(ValueError, match="Invalid model types"):
            metrics.run()
