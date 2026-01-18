from pathlib import Path
from typing import Any, override

import torch
from torch.utils.data import DataLoader, TensorDataset

from exp.trainers.base import ExperimentTrainer


class TrainerImpl(ExperimentTrainer):
    """Implementation for testing."""

    def __init__(
        self,
        data_size: int = 10,
        batch_size: int = 2,
        max_epochs: int = 1,
        max_steps_every_train: int | None = None,
    ):
        super().__init__()
        self._data_size = data_size
        self._batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_steps_every_train = max_steps_every_train
        self.batch_step_calls: list[tuple[Any, int]] = []

    @override
    def create_dataloader(self) -> DataLoader[Any]:
        dataset = TensorDataset(torch.randn(self._data_size, 4))
        return DataLoader(dataset, batch_size=self._batch_size)

    @override
    def create_optimizers(self) -> dict[str, Any]:
        return {}

    @override
    def batch_step(self, batch: Any, index: int) -> None:
        self.batch_step_calls.append((batch, index))


class TestExperimentTrainer:
    def test_setup_initializes_current_epoch(self):
        trainer = TrainerImpl()
        trainer.setup()

        assert trainer.current_epoch == 0

    def test_train_calls_batch_step_with_correct_indices(self):
        trainer = TrainerImpl(data_size=6, batch_size=2, max_epochs=1)
        trainer.setup()

        trainer.train()

        indices = [call[1] for call in trainer.batch_step_calls]
        assert indices == [0, 1, 2]

    def test_train_increments_global_steps(self):
        trainer = TrainerImpl(data_size=4, batch_size=2, max_epochs=2)
        trainer.setup()
        trainer.global_steps = 0

        trainer.train()

        assert trainer.global_steps == 4  # 2 batches * 2 epochs

    def test_train_updates_current_epoch(self):
        trainer = TrainerImpl(data_size=2, batch_size=2, max_epochs=3)
        trainer.setup()

        trainer.train()

        assert trainer.current_epoch == 2

    def test_train_stops_at_max_steps_every_train(self):
        trainer = TrainerImpl(
            data_size=10, batch_size=1, max_epochs=5, max_steps_every_train=3
        )
        trainer.setup()
        trainer.global_steps = 10

        trainer.train()

        assert trainer.global_steps == 13

    def test_save_and_load_state_preserves_global_steps(self, tmp_path: Path):
        state_path = tmp_path / "state"
        trainer = TrainerImpl()
        trainer.setup()
        trainer.global_steps = 42
        trainer.save_state(state_path)

        new_trainer = TrainerImpl()
        new_trainer.load_state(state_path)

        assert new_trainer.global_steps == 42
