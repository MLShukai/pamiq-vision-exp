from exp.tracking import ExperimentTracker


class TestExperimentTracker:
    def test_log_scalar(self, mocker):
        mock_task = mocker.Mock()
        mocker.patch("exp.tracking.Task.init", return_value=mock_task)

        tracker = ExperimentTracker(project_name="test", task_name="test")
        tracker.log_scalar("title", "series", 1.0, 1)

        mock_task.get_logger().report_scalar.assert_called_once_with(
            "title", "series", 1.0, 1
        )

    def test_log_text(self, mocker):
        mock_task = mocker.Mock()
        mocker.patch("exp.tracking.Task.init", return_value=mock_task)

        tracker = ExperimentTracker(project_name="test", task_name="test")
        tracker.log_text("hello")

        mock_task.get_logger().report_text.assert_called_once_with("hello")
