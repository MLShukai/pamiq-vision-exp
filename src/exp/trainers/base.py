from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, override

from pamiq_core.torch import TorchTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm


class ExperimentTrainer(TorchTrainer, ABC):
    """Base trainer class for vision experiments.

    This abstract base class provides common functionality for training
    vision models in the PAMIQ framework. It handles training loops,
    progress tracking, and integration with Aim for experiment tracking.
    """

    global_steps: int = 0
    """Total number of training steps across all epochs."""

    # Please override this parameters in constructor
    max_epochs: int = 1
    """Maximum number of epochs to train in a single training session."""

    max_steps_every_train: int | None = None
    """Maximum number of steps to train per session.

    If set, training stops after this many steps.
    """

    @abstractmethod
    def create_dataloader(self) -> DataLoader[Any]:
        """Create and return a DataLoader for training.

        Subclasses must implement this method to provide training data.

        Returns:
            DataLoader configured for the specific training task.
        """
        pass

    def batch_step(self, batch: Any, index: int) -> None:
        """Process a single batch of data.

        Subclasses should override this method to implement their training logic.

        Args:
            batch: A batch of data from the dataloader.
            index: The index of the current batch within the epoch.
        """
        pass

    @override
    def setup(self) -> None:
        super().setup()
        self.current_epoch = 0

    @override
    def train(self) -> None:
        dataloader = self.create_dataloader()
        start_global_step = self.global_steps
        for epoch in tqdm(range(self.max_epochs), desc="Epoch"):
            self.current_epoch = epoch
            for idx, batch in enumerate(tqdm(dataloader, leave=False, desc="Batch")):
                self.batch_step(batch, idx)
                self.global_steps += 1

                if self.max_steps_every_train is not None:
                    if (
                        self.global_steps - start_global_step
                        >= self.max_steps_every_train
                    ):
                        return

    @override
    def save_state(self, path: Path) -> None:
        """Save trainer state to disk."""
        super().save_state(path)
        path.mkdir(exist_ok=True)
        (path / "global_steps").write_text(str(self.global_steps), "utf-8")

    @override
    def load_state(self, path: Path) -> None:
        """Load trainer state from disk."""
        super().load_state(path)
        self.global_steps = int((path / "global_steps").read_text("utf-8"))
