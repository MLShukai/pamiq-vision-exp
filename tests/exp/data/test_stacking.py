import pytest
import torch

from exp.data.stacking import FrameStacker


class TestFrameStacker:
    def test_returns_none_until_enough_frames(self):
        stacker = FrameStacker(num_frames=3, stride=1)
        frame = torch.randn(3, 8, 8)

        assert stacker.push(frame) is None
        assert stacker.push(frame) is None
        result = stacker.push(frame)
        assert result is not None

    def test_output_shape(self):
        num_frames = 4
        stacker = FrameStacker(num_frames=num_frames, stride=1)
        C, H, W = 3, 16, 16
        frame = torch.randn(C, H, W)

        for _ in range(num_frames - 1):
            stacker.push(frame)

        result = stacker.push(frame)
        assert result is not None
        assert result.shape == (C, num_frames, H, W)

    def test_stride(self):
        stacker = FrameStacker(num_frames=3, stride=2)
        # history_size = (3-1)*2 + 1 = 5
        frames = [torch.full((1, 2, 2), float(i)) for i in range(5)]

        for i in range(4):
            assert stacker.push(frames[i]) is None

        result = stacker.push(frames[4])
        assert result is not None
        # Should pick frames at indices 0, 2, 4 (stride=2)
        assert torch.equal(result[:, 0, :, :], frames[0])
        assert torch.equal(result[:, 1, :, :], frames[2])
        assert torch.equal(result[:, 2, :, :], frames[4])

    def test_reset(self):
        stacker = FrameStacker(num_frames=2, stride=1)
        frame = torch.randn(3, 8, 8)

        stacker.push(frame)
        stacker.reset()

        # After reset, should need to accumulate frames again
        assert stacker.push(frame) is None
        assert stacker.push(frame) is not None

    def test_continuous_push_yields_output_after_fill(self):
        stacker = FrameStacker(num_frames=2, stride=1)
        frame = torch.randn(3, 8, 8)

        # First push returns None
        assert stacker.push(frame) is None
        # All subsequent pushes return stacked output
        for _ in range(5):
            assert stacker.push(frame) is not None

    @pytest.mark.parametrize(
        "num_frames, stride",
        [
            (0, 1),
            (-1, 1),
            (1, 0),
            (1, -1),
        ],
    )
    def test_invalid_parameters(self, num_frames: int, stride: int):
        with pytest.raises(ValueError):
            FrameStacker(num_frames=num_frames, stride=stride)

    def test_single_frame_no_stride(self):
        stacker = FrameStacker(num_frames=1, stride=1)
        frame = torch.randn(3, 8, 8)
        result = stacker.push(frame)
        assert result is not None
        assert result.shape == (3, 1, 8, 8)
