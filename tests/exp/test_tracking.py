from exp.tracking import ExperimentTracker


class TestExperimentTracker:
    def test_disabled_tracker_does_nothing(self):
        tracker = ExperimentTracker(enabled=False)
        # Should not raise
        tracker.log_scalar("title", "series", 1.0, 1)
        tracker.log_text("hello")

    def test_enabled_without_clearml_falls_back(self):
        """When ClearML is not configured, tracker falls back to disabled."""
        tracker = ExperimentTracker(
            enabled=True,
            project_name="test",
            task_name="test",
        )
        # ClearML may or may not be available; either way, should not raise
        tracker.log_scalar("title", "series", 1.0, 1)
        tracker.log_text("hello")
