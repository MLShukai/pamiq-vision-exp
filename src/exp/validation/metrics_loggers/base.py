from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset


class MetricsLogger(ABC):
    """Base class for metrics logging that processes datasets and models.

    This class provides a framework for logging metrics by accepting a
    dataset and models, then performing metric calculation and logging
    through the abstract run() method.
    """

    _dataset: Dataset[Any] | None = None

    def attach_dataset(self, dataset: Dataset[Any]) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset[Any]:
        if self._dataset is None:
            raise RuntimeError("Dataset not attached. Call attach_dataset() first.")
        return self._dataset

    _models: dict[str, nn.Module] | None = None

    def attach_models(self, models: dict[str, nn.Module]) -> None:
        self._models = models

    @property
    def models(self) -> dict[str, nn.Module]:
        if self._models is None:
            raise RuntimeError("Models not attached. Call attach_models() first.")
        return self._models

    _exp_cfg: DictConfig | ListConfig | None = None

    def attach_exp_cfg(self, cfg: DictConfig | ListConfig) -> None:
        self._exp_cfg = cfg

    @property
    def exp_cfg(self) -> DictConfig | ListConfig:
        if self._exp_cfg is None:
            raise RuntimeError(
                "Experiemnt config not attached. Call attach_exp_cfg() first."
            )
        return self._exp_cfg

    @abstractmethod
    def run(self) -> None:
        """Run metrics logging procedure."""
