from unittest.mock import MagicMock

import pytest
import torch

from exp.training.loop import TrainingLoop


def _make_loop(**overrides: object) -> TrainingLoop:
    """Create a TrainingLoop with mock dependencies."""
    defaults: dict[str, object] = {
        "frame_loader": MagicMock(),
        "frame_stacker": MagicMock(),
        "buffer": MagicMock(),
        "training_logic": MagicMock(),
        "checkpoint_manager": MagicMock(),
        "trigger_every_n_frames": 5,
        "batch_size": 4,
        "num_epochs": 1,
        "checkpoint_interval_seconds": 300.0,
        "device": torch.device("cpu"),
    }
    defaults.update(overrides)
    return TrainingLoop(**defaults)  # type: ignore[arg-type]


def _patch_time_no_checkpoint(mocker, num_frames: int):
    """Patch time so no timed checkpoints fire (all intervals < threshold).

    Returns enough monotonic values for: start + num_frames iterations +
    final.
    """
    mock_time = mocker.patch("exp.training.loop.time")
    # All times close together so checkpoint interval never triggers
    times = [float(i) for i in range(num_frames + 3)]
    mock_time.monotonic = MagicMock(side_effect=times)
    return mock_time


class TestTrainingLoop:
    def test_validation_trigger_every_n_frames(self):
        with pytest.raises(ValueError, match="trigger_every_n_frames"):
            _make_loop(trigger_every_n_frames=0)
        with pytest.raises(ValueError, match="trigger_every_n_frames"):
            _make_loop(trigger_every_n_frames=-1)

    def test_validation_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            _make_loop(batch_size=0)

    def test_validation_num_epochs(self):
        with pytest.raises(ValueError, match="num_epochs"):
            _make_loop(num_epochs=0)

    def test_frames_flow_to_buffer(self, mocker):
        frame_loader = MagicMock()
        frames = [torch.randn(3, 8, 8) for _ in range(4)]
        frame_loader.__iter__ = MagicMock(return_value=iter(frames))

        stacker = MagicMock()
        stacked_tensor = torch.randn(3, 2, 8, 8)
        # Return None for first frame, tensor for the rest
        stacker.push = MagicMock(
            side_effect=[None, stacked_tensor, stacked_tensor, stacked_tensor]
        )

        buffer = MagicMock()
        buffer.__len__ = MagicMock(return_value=1)

        _patch_time_no_checkpoint(mocker, 4)

        loop = _make_loop(
            frame_loader=frame_loader,
            frame_stacker=stacker,
            buffer=buffer,
            trigger_every_n_frames=100,  # high so trigger doesn't fire
        )
        loop.run({})

        assert buffer.add.call_count == 3

    def test_trigger_fires_at_correct_interval(self, mocker):
        frame_loader = MagicMock()
        frames = [torch.randn(3, 8, 8) for _ in range(6)]
        frame_loader.__iter__ = MagicMock(return_value=iter(frames))

        stacker = MagicMock()
        stacked = torch.randn(3, 2, 8, 8)
        stacker.push = MagicMock(return_value=stacked)

        buffer = MagicMock()
        buffer.__len__ = MagicMock(return_value=10)
        buffer.get_data = MagicMock(return_value=torch.randn(10, 3, 2, 8, 8))

        training_logic = MagicMock()
        result_mock = MagicMock()
        result_mock.loss.item.return_value = 0.5
        result_mock.metrics = {}
        training_logic.train_step_from_batch = MagicMock(return_value=result_mock)

        _patch_time_no_checkpoint(mocker, 6)

        loop = _make_loop(
            frame_loader=frame_loader,
            frame_stacker=stacker,
            buffer=buffer,
            training_logic=training_logic,
            trigger_every_n_frames=3,
            batch_size=4,
            num_epochs=1,
        )
        loop.run({})

        # 6 frames, trigger every 3 => 2 triggers
        assert buffer.get_data.call_count == 2

    def test_checkpoint_saved_at_time_interval(self, mocker):
        frame_loader = MagicMock()
        frames = [torch.randn(3, 8, 8) for _ in range(3)]
        frame_loader.__iter__ = MagicMock(return_value=iter(frames))

        stacker = MagicMock()
        stacker.push = MagicMock(return_value=torch.randn(3, 2, 8, 8))

        buffer = MagicMock()
        buffer.__len__ = MagicMock(return_value=1)

        checkpoint_manager = MagicMock()

        mock_time = mocker.patch("exp.training.loop.time")
        # Sequence of monotonic() calls:
        # 1. start_time = 0.0
        # 2. after frame 1: current_time = 100.0 (< 300, no ckpt)
        # 3. after frame 2: current_time = 400.0 (>= 300, ckpt! last_ckpt=400)
        # 4. after frame 3: current_time = 500.0 (< 300 since 400, no ckpt)
        # 5. final checkpoint elapsed
        mock_time.monotonic = MagicMock(side_effect=[0.0, 100.0, 400.0, 500.0, 600.0])

        loop = _make_loop(
            frame_loader=frame_loader,
            frame_stacker=stacker,
            buffer=buffer,
            checkpoint_manager=checkpoint_manager,
            trigger_every_n_frames=100,
            checkpoint_interval_seconds=300.0,
        )

        models = {"encoder": MagicMock()}
        loop.run(models)

        # 1 timed checkpoint + 1 final checkpoint = 2 saves
        assert checkpoint_manager.save.call_count == 2

    def test_final_checkpoint_always_saved(self, mocker):
        frame_loader = MagicMock()
        frames = [torch.randn(3, 8, 8) for _ in range(2)]
        frame_loader.__iter__ = MagicMock(return_value=iter(frames))

        stacker = MagicMock()
        stacker.push = MagicMock(return_value=torch.randn(3, 2, 8, 8))

        buffer = MagicMock()
        buffer.__len__ = MagicMock(return_value=1)

        checkpoint_manager = MagicMock()

        mock_time = mocker.patch("exp.training.loop.time")
        # No timed checkpoint triggers (all intervals < 300)
        mock_time.monotonic = MagicMock(side_effect=[0.0, 10.0, 20.0, 30.0])

        loop = _make_loop(
            frame_loader=frame_loader,
            frame_stacker=stacker,
            buffer=buffer,
            checkpoint_manager=checkpoint_manager,
            trigger_every_n_frames=100,
            checkpoint_interval_seconds=300.0,
        )

        models = {"encoder": MagicMock()}
        loop.run(models)

        # Only the final checkpoint
        assert checkpoint_manager.save.call_count == 1

    def test_learn_uses_dataloader_from_buffer(self, mocker):
        frame_loader = MagicMock()
        frames = [torch.randn(3, 8, 8) for _ in range(3)]
        frame_loader.__iter__ = MagicMock(return_value=iter(frames))

        stacker = MagicMock()
        stacker.push = MagicMock(return_value=torch.randn(3, 2, 8, 8))

        buffer = MagicMock()
        buffer.__len__ = MagicMock(return_value=8)
        buffer.get_data = MagicMock(return_value=torch.randn(8, 3, 2, 8, 8))

        training_logic = MagicMock()
        result_mock = MagicMock()
        result_mock.loss.item.return_value = 0.5
        result_mock.metrics = {}
        training_logic.train_step_from_batch = MagicMock(return_value=result_mock)

        _patch_time_no_checkpoint(mocker, 3)

        loop = _make_loop(
            frame_loader=frame_loader,
            frame_stacker=stacker,
            buffer=buffer,
            training_logic=training_logic,
            trigger_every_n_frames=3,
            batch_size=4,
            num_epochs=1,
        )
        loop.run({})

        # get_data is called to create DataLoader
        buffer.get_data.assert_called()
        # DataLoader(8 items, batch_size=4) => 2 batches, 1 epoch => 2 calls
        assert training_logic.train_step_from_batch.call_count == 2
