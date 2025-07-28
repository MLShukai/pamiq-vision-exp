from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from aim.storage.types import AimObjectDict
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from exp.aim_utils import get_global_run


class MetricsLogger(ABC):
    """Base class for metrics logging."""

    def __init__(self, metrics_type: str) -> None:
        super().__init__()
        self.default_aim_context: AimObjectDict = {
            "namespace": "static_metrics",
            "metrics_type": metrics_type,
        }

    _dataset: Dataset[Any] | None = None

    def attach_dataset(self, dataset: Dataset[Any]) -> None:
        """Attach dataset for metrics computation."""
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset[Any]:
        if self._dataset is None:
            raise RuntimeError("Dataset not attached. Call attach_dataset() first.")
        return self._dataset

    _models: Mapping[str, nn.Module] | None = None

    def attach_models(self, models: Mapping[str, nn.Module]) -> None:
        """Attach models to evaluate."""
        self._models = models

    @property
    def models(self) -> Mapping[str, nn.Module]:
        if self._models is None:
            raise RuntimeError("Models not attached. Call attach_models() first.")
        return self._models

    _exp_cfg: DictConfig | ListConfig | None = None

    def attach_exp_cfg(self, cfg: DictConfig | ListConfig) -> None:
        """Attach experiment configuration."""
        self._exp_cfg = cfg

    @property
    def exp_cfg(self) -> DictConfig | ListConfig:
        if self._exp_cfg is None:
            raise RuntimeError(
                "Experiemnt config not attached. Call attach_exp_cfg() first."
            )
        return self._exp_cfg

    @torch.inference_mode()
    def run(self) -> None:
        """Run metrics computation on the dataset."""
        self._setup()

        dataloader = self.create_dataloader()
        for idx, batch in enumerate(tqdm(dataloader)):
            self.batch_step(batch, idx)

        self.teardown()

    def _setup(self) -> None:
        if (aim_run := get_global_run()) is None:
            raise ValueError(
                "Aim run not initialized. Please set global aim run before calling JEPAMetrics."
            )
        self.aim_run = aim_run

        self.setup()

    def setup(self) -> None:
        """Optional setup before metrics computation."""
        pass

    @abstractmethod
    def create_dataloader(self) -> DataLoader[Any]:
        """Create dataloader for the attached dataset."""
        pass

    def batch_step(self, batch: Any, index: int) -> None:
        """Process a single batch.

        Override to implement metrics.
        """
        pass

    def teardown(self) -> None:
        """Optional cleanup after metrics computation."""
        pass
