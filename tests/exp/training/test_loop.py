from pathlib import Path

import pytest
import torch
from torch import nn

from exp.data.buffer import FIFOReplayBuffer
from exp.data.stacking import FrameStacker
from exp.training.checkpoint import CheckpointManager
from exp.training.loop import TrainingLoop
from tests.helpers import TrainingLogicImpl


def _make_loop(tmp_path: Path, **overrides: object) -> TrainingLoop:
    """Create a TrainingLoop with real dependencies."""
    defaults: dict[str, object] = {
        "frame_loader": [],
        "frame_stacker": FrameStacker(num_frames=2, stride=1),
        "buffer": FIFOReplayBuffer(max_size=100),
        "training_logic": TrainingLogicImpl(),
        "checkpoint_manager": CheckpointManager(
            log_dir=tmp_path, experiment_name="test"
        ),
        "trigger_every_n_frames": 5,
        "batch_size": 4,
        "num_epochs": 1,
        "checkpoint_interval_seconds": 300.0,
        "device": torch.device("cpu"),
    }
    defaults.update(overrides)
    return TrainingLoop(**defaults)  # type: ignore[arg-type]


def _patch_time_no_checkpoint(mocker, num_stacked_frames: int):
    """Patch time so no timed checkpoints fire.

    Returns enough monotonic values for: start + num_stacked_frames + final.
    """
    mock_time = mocker.patch("exp.training.loop.time")
    times = [float(i) for i in range(num_stacked_frames + 2)]
    mock_time.monotonic = mocker.Mock(side_effect=times)
    return mock_time


class TestTrainingLoop:
    def test_validation_trigger_every_n_frames(self, tmp_path: Path):
        with pytest.raises(ValueError, match="trigger_every_n_frames"):
            _make_loop(tmp_path, trigger_every_n_frames=0)
        with pytest.raises(ValueError, match="trigger_every_n_frames"):
            _make_loop(tmp_path, trigger_every_n_frames=-1)

    def test_validation_batch_size(self, tmp_path: Path):
        with pytest.raises(ValueError, match="batch_size"):
            _make_loop(tmp_path, batch_size=0)

    def test_validation_num_epochs(self, tmp_path: Path):
        with pytest.raises(ValueError, match="num_epochs"):
            _make_loop(tmp_path, num_epochs=0)

    def test_frames_flow_to_buffer(self, mocker, tmp_path: Path):
        # 4 frames with FrameStacker(num_frames=2): first None + 3 stacked
        frames = [torch.randn(3, 8, 8) for _ in range(4)]
        buffer = FIFOReplayBuffer(max_size=100)

        _patch_time_no_checkpoint(mocker, 3)

        loop = _make_loop(
            tmp_path,
            frame_loader=frames,
            buffer=buffer,
            trigger_every_n_frames=100,
        )
        loop.run({})

        assert len(buffer) == 3

    def test_trigger_fires_at_correct_interval(self, mocker, tmp_path: Path):
        # 7 frames with FrameStacker(num_frames=2): first None + 6 stacked
        # trigger_every_n_frames=3 => 2 triggers
        frames = [torch.randn(3, 8, 8) for _ in range(7)]
        buffer = FIFOReplayBuffer(max_size=100)
        training_logic = TrainingLogicImpl()

        _patch_time_no_checkpoint(mocker, 6)

        loop = _make_loop(
            tmp_path,
            frame_loader=frames,
            buffer=buffer,
            training_logic=training_logic,
            trigger_every_n_frames=3,
            batch_size=2,
            num_epochs=1,
        )
        loop.run({})

        # 1st trigger: 3 items, batch_size=2 => 2 batches
        # 2nd trigger: 6 items, batch_size=2 => 3 batches
        # total: 5 train steps
        assert training_logic.call_count == 5

    def test_checkpoint_saved_at_time_interval(self, mocker, tmp_path: Path):
        # 4 frames with FrameStacker(num_frames=2): first None + 3 stacked
        frames = [torch.randn(3, 8, 8) for _ in range(4)]
        buffer = FIFOReplayBuffer(max_size=100)
        checkpoint_manager = CheckpointManager(log_dir=tmp_path, experiment_name="test")

        mock_time = mocker.patch("exp.training.loop.time")
        # monotonic calls: start=0.0, frame2=100.0, frame3=400.0(ckpt), frame4=500.0, final=600.0
        mock_time.monotonic = mocker.Mock(side_effect=[0.0, 100.0, 400.0, 500.0, 600.0])

        loop = _make_loop(
            tmp_path,
            frame_loader=frames,
            buffer=buffer,
            checkpoint_manager=checkpoint_manager,
            trigger_every_n_frames=100,
            checkpoint_interval_seconds=300.0,
        )

        models = {"encoder": nn.Linear(4, 4)}
        loop.run(models)

        # 1 timed checkpoint + 1 final checkpoint = 2 saves
        ckpts = list(checkpoint_manager.checkpoint_dir.glob("*.ckpt"))
        assert len(ckpts) == 2

    def test_final_checkpoint_always_saved(self, mocker, tmp_path: Path):
        # 3 frames with FrameStacker(num_frames=2): first None + 2 stacked
        frames = [torch.randn(3, 8, 8) for _ in range(3)]
        buffer = FIFOReplayBuffer(max_size=100)
        checkpoint_manager = CheckpointManager(log_dir=tmp_path, experiment_name="test")

        mock_time = mocker.patch("exp.training.loop.time")
        # monotonic calls: start=0.0, frame2=10.0, frame3=20.0, final=30.0
        mock_time.monotonic = mocker.Mock(side_effect=[0.0, 10.0, 20.0, 30.0])

        loop = _make_loop(
            tmp_path,
            frame_loader=frames,
            buffer=buffer,
            checkpoint_manager=checkpoint_manager,
            trigger_every_n_frames=100,
            checkpoint_interval_seconds=300.0,
        )

        models = {"encoder": nn.Linear(4, 4)}
        loop.run(models)

        # Only the final checkpoint
        ckpts = list(checkpoint_manager.checkpoint_dir.glob("*.ckpt"))
        assert len(ckpts) == 1

    def test_learn_uses_dataloader_from_buffer(self, mocker, tmp_path: Path):
        # 9 frames with FrameStacker(num_frames=2): first None + 8 stacked
        # trigger_every_n_frames=8 => 1 trigger with 8 items
        frames = [torch.randn(3, 8, 8) for _ in range(9)]
        buffer = FIFOReplayBuffer(max_size=100)
        training_logic = TrainingLogicImpl()

        _patch_time_no_checkpoint(mocker, 8)

        loop = _make_loop(
            tmp_path,
            frame_loader=frames,
            buffer=buffer,
            training_logic=training_logic,
            trigger_every_n_frames=8,
            batch_size=4,
            num_epochs=1,
        )
        loop.run({})

        # DataLoader(8 items, batch_size=4) => 2 batches
        assert training_logic.call_count == 2
