from exp.tracking import ExperimentTracker


class TestExperimentTracker:
    def test_tracker_falls_back_when_clearml_unavailable(self):
        """When ClearML is not available, tracker falls back gracefully."""
        tracker = ExperimentTracker(project_name="test", task_name="test")
        # Should not raise
        tracker.log_scalar("title", "series", 1.0, 1)
        tracker.log_text("hello")
