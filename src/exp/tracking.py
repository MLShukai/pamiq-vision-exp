"""Experiment tracking with ClearML."""

from clearml import Task


class ExperimentTracker:
    """ClearML experiment tracker."""

    def __init__(
        self,
        project_name: str = "",
        task_name: str = "",
    ) -> None:
        self._task = Task.init(
            project_name=project_name,
            task_name=task_name,
        )
        self._logger = self._task.get_logger()

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        """Log a scalar value."""
        self._logger.report_scalar(title, series, value, iteration)

    def log_text(self, message: str) -> None:
        """Log a text message."""
        self._logger.report_text(message)
