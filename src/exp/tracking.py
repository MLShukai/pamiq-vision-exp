"""Experiment tracking with ClearML."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """ClearML experiment tracker.

    Wraps ClearML Task and Logger. When ClearML is unavailable, all
    logging calls are silently skipped.
    """

    def __init__(
        self,
        project_name: str = "",
        task_name: str = "",
    ) -> None:
        self._task: Any = None
        self._logger: Any = None

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

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        """Log a scalar value."""
        if self._logger is not None:
            self._logger.report_scalar(title, series, value, iteration)

    def log_text(self, message: str) -> None:
        """Log a text message."""
        if self._logger is not None:
            self._logger.report_text(message)
