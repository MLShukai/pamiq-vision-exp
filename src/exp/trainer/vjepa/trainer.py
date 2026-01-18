"""V-JEPA Trainer for PAMIQ Core integration."""

import itertools
from collections.abc import Callable
from functools import partial
from typing import override

import torch
from pamiq_core.data import DataUser
from pamiq_core.data.impls import RandomReplacementBuffer
from pamiq_core.torch import OptimizersSetup, TorchTrainingModel, get_device
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from exp.models.vjepa import Encoder, Predictor

from ..base import ExperimentTrainer
from .logic import VJEPATrainingLogic

OPTIMIZER_NAME = "vjepa_optimizer"


class VJEPATrainer(ExperimentTrainer):
    """Trainer for Video Joint Embedding Predictive Architecture (V-JEPA).

    This trainer integrates VJEPATrainingLogic with the PAMIQ Core
    framework, handling data management, model synchronization, and
    state persistence.

    The actual training logic (loss computation, EMA updates) is
    delegated to the VJEPATrainingLogic class.
    """

    @override
    def __init__(
        self,
        partial_optimizer: partial[Optimizer],
        collate_fn: Callable[
            [list[tuple[Tensor, ...] | Tensor]], tuple[Tensor, Tensor, Tensor]
        ],
        context_encoder_name: str = "context_encoder",
        target_encoder_name: str = "target_encoder",
        predictor_name: str = "predictor_name",
        data_user_name: str = "video",
        log_prefix: str = "vjepa",
        ema_momentum: float = 0.998,
        batch_size: int = 1,
        max_epochs: int = 1,
        min_buffer_size: int = 0,
        min_new_data_count: int = 0,
        max_steps_every_train: int | None = None,
    ) -> None:
        """Initialize the V-JEPA trainer.

        Args:
            partial_optimizer: Partially configured optimizer to be used with
                the model parameters.
            collate_fn: Collator function for sampling input data, encoder mask
                and predictor target. Typically VideoMultiBlockMaskCollator.
            context_encoder_name: Name of the context encoder model to retrieve
                from the model registry.
            target_encoder_name: Name of the target encoder model to retrieve
                from the model registry.
            predictor_name: Name of the predictor model to retrieve from the
                model registry.
            data_user_name: Name of the data user providing training data.
            log_prefix: Prefix for training metrics in logging.
            ema_momentum: Momentum coefficient for updating the target encoder
                from the context encoder. Higher values mean slower updates.
                Default 0.998 based on V-JEPA paper.
            batch_size: Data sample size for 1 step.
            max_epochs: Maximum number of epochs to train per training session.
            min_buffer_size: Minimum buffer size required before training starts.
            min_new_data_count: Minimum number of new data points required for training.
            max_steps_every_train: Maximum number of steps to train per session.
                If set, training stops after this many steps.
        """
        super().__init__(data_user_name, min_buffer_size, min_new_data_count)

        self._data_user_name = data_user_name
        self._partial_optimizer = partial_optimizer
        self._partial_dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self._context_encoder_name = context_encoder_name
        self._target_encoder_name = target_encoder_name
        self._predictor_name = predictor_name
        self._log_prefix = log_prefix
        self._ema_momentum = ema_momentum
        self.max_epochs = max_epochs
        self.max_steps_every_train = max_steps_every_train

    @override
    def on_data_users_attached(self) -> None:
        """Set up data user references when they are attached to the
        trainer."""
        super().on_data_users_attached()
        self._data_user: DataUser[list[Tensor]] = self.get_data_user(
            self._data_user_name
        )

    @override
    def on_training_models_attached(self) -> None:
        """Set up model references when they are attached to the trainer."""
        super().on_training_models_attached()
        self._context_encoder: TorchTrainingModel[Encoder] = (
            self.get_torch_training_model(self._context_encoder_name, Encoder)
        )
        self._target_encoder: TorchTrainingModel[Encoder] = (
            self.get_torch_training_model(self._target_encoder_name, Encoder)
        )
        self._predictor: TorchTrainingModel[Predictor] = self.get_torch_training_model(
            self._predictor_name, Predictor
        )

    @override
    def create_optimizers(self) -> OptimizersSetup:
        """Create optimizers for V-JEPA training.

        Creates an optimizer that updates both the context encoder and predictor
        parameters. The target encoder is updated separately through exponential
        moving average and is not directly optimized.

        Returns:
            Dictionary mapping optimizer name to configured optimizer instance.
        """
        return {
            OPTIMIZER_NAME: self._partial_optimizer(
                itertools.chain(
                    self._context_encoder.model.parameters(),
                    self._predictor.model.parameters(),
                )
            )
        }

    @override
    def setup(self) -> None:
        """Set up trainer state before training begins."""
        super().setup()
        self._device = get_device(self._context_encoder.model)

        # Initialize training logic with models and optimizer
        self._training_logic = VJEPATrainingLogic(
            context_encoder=self._context_encoder.model,
            target_encoder=self._target_encoder.model,
            predictor=self._predictor.model,
            optimizer=self.optimizers[OPTIMIZER_NAME],
            ema_momentum=self._ema_momentum,
        )

    @override
    def create_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        """Create a DataLoader for training.

        Returns:
            DataLoader configured with video data and collation function.
        """
        dataset = TensorDataset(torch.stack(self._data_user.get_data()))
        return self._partial_dataloader(dataset=dataset)

    @override
    def batch_step(self, batch: tuple[Tensor, Tensor, Tensor], index: int) -> None:
        """Process a single batch of data.

        Args:
            batch: Tuple of (videos, encoder_masks, predictor_targets).
            index: The index of the current batch within the epoch.
        """
        videos, masks_for_context_encoder, targets_for_predictor = batch

        result = self._training_logic.train_step(
            videos.to(self._device),
            masks_for_context_encoder.to(self._device),
            targets_for_predictor.to(self._device),
        )

        # Log metrics
        _ = {
            f"{self._log_prefix}/loss": result.loss.item(),
            f"{self._log_prefix}/target_encoder_latent_std": (
                result.latent_target.std(0).mean().item()
            ),
            f"{self._log_prefix}/context_encoder_latent_std": (
                result.latent_context.std(0).mean().item()
            ),
        }

    @staticmethod
    def create_buffer(
        batch_size: int,
        iteration_count: int,
        expected_survival_length: int,
    ) -> RandomReplacementBuffer[Tensor]:
        """Create data buffer for this trainer.

        Args:
            batch_size: Batch size for training.
            iteration_count: Number of iterations per training session.
            expected_survival_length: Expected survival length for buffer items.

        Returns:
            Configured RandomReplacementBuffer for video data.
        """
        return RandomReplacementBuffer[Tensor](
            max_size=iteration_count * batch_size,
            expected_survival_length=expected_survival_length,
        )
