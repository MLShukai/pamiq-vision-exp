"""Experiment tracking with ClearML."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Optional ClearML experiment tracker.

    Wraps ClearML Task and Logger. When disabled or ClearML is
    unavailable, all logging calls are silently skipped.
    """

    def __init__(
        self,
        enabled: bool = False,
        project_name: str = "",
        task_name: str = "",
    ) -> None:
        self._enabled = enabled
        self._task: Any = None
        self._logger: Any = None

        if enabled:
            try:
                from clearml import Task

                self._task = Task.init(
                    project_name=project_name,
                    task_name=task_name,
                )
                self._logger = self._task.get_logger()
                logger.info("ClearML tracking enabled")
            except Exception as e:
                logger.warning(f"ClearML initialization failed: {e}")
                self._enabled = False

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        """Log a scalar value."""
        if self._enabled and self._logger is not None:
            self._logger.report_scalar(title, series, value, iteration)

    def log_text(self, message: str) -> None:
        """Log a text message."""
        if self._enabled and self._logger is not None:
            self._logger.report_text(message)
