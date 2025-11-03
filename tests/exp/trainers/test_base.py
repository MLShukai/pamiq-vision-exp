from pathlib import Path
from typing import Any, override
from unittest.mock import MagicMock

import pytest
import torch
from pamiq_core.torch import OptimizersSetup
from pytest_mock import MockerFixture
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from exp.trainers.base import ExperimentTrainer


class ConcreteTrainer(ExperimentTrainer):
    """Concrete implementation of ExperimentTrainer for testing."""

    def __init__(self, **kwargs):
        # TorchTrainer doesn't take these args, so we need to pop them
        kwargs.pop("data_user_name", None)
        kwargs.pop("min_buffer_size", None)
        kwargs.pop("min_new_data_count", None)
        super().__init__(**kwargs)
        self.create_dataloader_called = False
        self.batch_step_called = False
        self.batch_step_count = 0
        self.last_batch = None
        self.last_index = None

    @override
    def create_dataloader(self) -> DataLoader[Any]:
        """Create a simple dataloader for testing."""
        self.create_dataloader_called = True
        dataset = TensorDataset(torch.randn(10, 3, 32, 32))
        return DataLoader(dataset, batch_size=2, shuffle=False)

    @override
    def batch_step(self, batch: Any, index: int) -> None:
        """Track batch step calls for testing."""
        self.batch_step_called = True
        self.batch_step_count += 1
        self.last_batch = batch
        self.last_index = index

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create dummy optimizer for testing."""
        # Create a dummy parameter to optimize
        dummy_param = torch.nn.Parameter(torch.randn(1))
        return {"test_optimizer": SGD([dummy_param], lr=0.01)}


class TestExperimentTrainer:
    """Test suite for ExperimentTrainer base class."""

    @pytest.fixture
    def mock_aim_run(self, mocker: MockerFixture):
        """Mock Aim run for testing."""
        mock_run = MagicMock()
        mocker.patch("exp.trainers.base.get_global_run", return_value=mock_run)
        return mock_run

    @pytest.fixture
    def trainer(self):
        """Create a concrete trainer instance for testing."""
        return ConcreteTrainer()

    def test_initialization(self, trainer: ConcreteTrainer):
        """Test trainer initialization with default values."""
        assert trainer.global_steps == 0
        assert trainer.max_epochs == 1
        assert trainer.max_steps_every_train is None
        assert trainer.trainer_name == "experiment"
        assert not trainer.create_dataloader_called
        assert not trainer.batch_step_called

    def test_default_aim_context(self, trainer: ConcreteTrainer):
        """Test default Aim context property."""
        context = trainer.default_aim_context
        assert context == {"trainer": "experiment"}

        # Test with custom trainer name
        trainer.trainer_name = "custom"
        assert trainer.default_aim_context == {"trainer": "custom"}

    def test_setup_with_aim_run(self, trainer: ConcreteTrainer, mock_aim_run):
        """Test setup method with valid Aim run."""
        trainer.setup()

        assert trainer.current_epoch == 0
        assert trainer.aim_run is mock_aim_run

    def test_setup_without_aim_run(
        self, trainer: ConcreteTrainer, mocker: MockerFixture
    ):
        """Test setup method raises error when no Aim run is found."""
        mocker.patch("exp.trainers.base.get_global_run", return_value=None)

        with pytest.raises(ValueError, match="No global Aim run found"):
            trainer.setup()

    def test_train_basic(self, trainer: ConcreteTrainer, mock_aim_run):
        """Test basic training loop."""
        trainer.max_epochs = 2
        trainer.setup()
        trainer.train()

        assert trainer.create_dataloader_called
        assert trainer.batch_step_called
        assert trainer.batch_step_count == 10  # 5 batches * 2 epochs
        assert trainer.global_steps == 10
        assert trainer.current_epoch == 1  # Last completed epoch

    def test_train_with_max_steps(self, trainer: ConcreteTrainer, mock_aim_run):
        """Test training with max_steps_every_train limit."""
        trainer.max_epochs = 5
        trainer.max_steps_every_train = 3
        trainer.setup()
        trainer.train()

        assert trainer.batch_step_count == 3
        assert trainer.global_steps == 3

    def test_train_continuation(self, trainer: ConcreteTrainer, mock_aim_run):
        """Test training continuation with existing global_steps."""
        trainer.global_steps = 5
        trainer.max_epochs = 1
        trainer.max_steps_every_train = 3
        trainer.setup()
        trainer.train()

        # Should only train 3 more steps
        assert trainer.batch_step_count == 3
        assert trainer.global_steps == 8

    def test_save_and_load_state(self, trainer: ConcreteTrainer, tmp_path: Path):
        """Test saving and loading trainer state."""
        # Set some state
        trainer.global_steps = 42

        # Save state
        save_path = tmp_path / "trainer_state"
        trainer.save_state(save_path)

        # Verify files were created
        assert save_path.exists()
        assert (save_path / "global_steps").exists()
        assert (save_path / "global_steps").read_text() == "42"

        # Create new trainer and load state
        new_trainer = ConcreteTrainer()
        new_trainer.global_steps = 0
        new_trainer.load_state(save_path)

        # Verify state was loaded
        assert new_trainer.global_steps == 42

    def test_batch_step_arguments(self, trainer: ConcreteTrainer, mock_aim_run):
        """Test that batch_step receives correct arguments."""
        trainer.max_epochs = 1
        trainer.setup()
        trainer.train()

        # Check last batch and index
        assert trainer.last_batch is not None
        # TensorDataset returns a list with one element when using DataLoader
        assert isinstance(trainer.last_batch, list)
        assert (
            len(trainer.last_batch) == 1
        )  # TensorDataset returns list with one tensor
        assert trainer.last_batch[0].shape == torch.Size([2, 3, 32, 32])
        assert trainer.last_index == 4  # Last batch index (0-4 for 5 batches)

    def test_custom_trainer_name(self, mock_aim_run):
        """Test trainer with custom name."""
        trainer = ConcreteTrainer()
        trainer.trainer_name = "my_custom_trainer"

        assert trainer.default_aim_context == {"trainer": "my_custom_trainer"}
