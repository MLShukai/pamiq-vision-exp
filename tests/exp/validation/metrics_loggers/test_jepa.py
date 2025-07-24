from typing import override

import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import Dataset

from exp.models import ModelName
from exp.models.jepa import Encoder, Predictor
from exp.validation.metrics_loggers.jepa import JEPAMetrics


class DummyDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self):
        return self.size

    @override
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32)


class TestJEPAMetrics:
    """Test cases for JEPAMetrics class."""

    def test_run_success(self, mocker: MockerFixture):
        """Test successful execution of run method."""
        # Mock dependencies
        mock_aim_run = mocker.Mock()
        mocker.patch(
            "exp.validation.metrics_loggers.jepa.get_global_run",
            return_value=mock_aim_run,
        )

        # Mock models
        mock_context_encoder = mocker.Mock(spec=Encoder)
        mock_target_encoder = mocker.Mock(spec=Encoder)
        mock_predictor = mocker.Mock(spec=Predictor)

        # Mock device
        mocker.patch(
            "exp.validation.metrics_loggers.jepa.get_device",
            return_value=torch.device("cpu"),
        )

        # Mock Hydra instantiate for collate_fn
        mock_collate_fn = mocker.Mock(side_effect=lambda batch: (torch.stack(batch),))
        mocker.patch("hydra.utils.instantiate", return_value=mock_collate_fn)

        # Mock JEPATrainer.compute_loss
        loss_per_data = [torch.tensor(0.5), torch.tensor(0.3)]
        mock_loss_dict = dict(loss=torch.tensor(0.4), loss_per_data=loss_per_data)
        mocker.patch(
            "exp.validation.metrics_loggers.jepa.JEPATrainer.compute_loss",
            return_value=mock_loss_dict,
        )

        # Create metrics instance
        metrics = JEPAMetrics(batch_size=2, log_prefix="test_jepa")
        metrics.attach_dataset(DummyDataset(size=4))
        metrics.attach_exp_cfg(mocker.MagicMock())
        metrics.attach_models(
            {
                ModelName.JEPA_CONTEXT_ENCODER: mock_context_encoder,
                ModelName.JEPA_TARGET_ENCODER: mock_target_encoder,
                ModelName.JEPA_PREDICTOR: mock_predictor,
            }
        )

        # Run the method
        metrics.run()

        # Verify aim_run.track was called for each sample
        assert mock_aim_run.track.call_count == 4  # 2 batches * 2 samples per batch

        # Verify track calls - check each call separately to handle floating point precision
        calls = mock_aim_run.track.call_args_list

        # Check first call
        assert calls[0][0][0] == pytest.approx(0.5)
        assert calls[0][0][1] == "loss"
        assert calls[0][1]["step"] == 0
        assert calls[0][1]["context"] == {
            "namespace": "metrics",
            "metrics_type": "test_jepa",
        }

        # Check second call
        assert calls[1][0][0] == pytest.approx(0.3)
        assert calls[1][0][1] == "loss"
        assert calls[1][1]["step"] == 1
        assert calls[1][1]["context"] == {
            "namespace": "metrics",
            "metrics_type": "test_jepa",
        }
