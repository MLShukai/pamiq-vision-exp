from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from pamiq_vision_exp.envs.video_frame_samplers import RandomVideoFrameSampler


class TestRandomVideoFrameSampler:
    @pytest.fixture
    def video_and_info(self, tmp_path: Path):
        """Create test directory and video files for testing.

        Args:
            tmp_path: Pytest fixture providing a temporary directory

        Returns:
            Tuple of (video directory Path, list of video info dictionaries)
        """
        # Create test video files
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        video_info = []

        # Video 1: Red frames (10 frames)
        video1_path = video_dir / "video1.mp4"
        self._create_test_video(video1_path, (64, 64), 10, color=(0, 0, 255))
        video_info.append({"path": video1_path, "frames": 10})

        # Video 2: Green frames (5 frames)
        video2_path = video_dir / "video2.mp4"
        self._create_test_video(video2_path, (64, 64), 5, color=(0, 255, 0))
        video_info.append({"path": video2_path, "frames": 5})

        # Video 3: Blue frames (15 frames)
        video3_path = video_dir / "video3.mp4"
        self._create_test_video(video3_path, (64, 64), 15, color=(255, 0, 0))
        video_info.append({"path": video3_path, "frames": 15})

        return video_dir, video_info

    def _create_test_video(self, filename, size, num_frames, color=(0, 0, 255)):
        """Create a test video file with solid color frames.

        Args:
            filename: Path to the video file to create
            size: Tuple of (width, height) for the video frames
            num_frames: Number of frames to include in the video
            color: BGR color tuple for the frames
        """
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video = cv2.VideoWriter(str(filename), fourcc, 30.0, size)

        for _ in range(num_frames):
            # Create solid color frame
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:] = color
            video.write(frame)

        video.release()

    def test_init_with_valid_data(self, video_and_info):
        """Test initialization with valid video files."""
        video_dir, video_info = video_and_info

        # Initialize sampler
        sampler = RandomVideoFrameSampler(folder=video_dir, extensions=["mp4"])

        # Check video file count
        assert len(sampler.video_files) == len(video_info)

        # Check frame counts and probabilities
        total_frames = sum(video["frames"] for video in video_info)
        expected_probs = [video["frames"] / total_frames for video in video_info]

        # Check frame counts (sorting since order is not guaranteed)
        assert sorted(sampler.frame_counts) == sorted(
            [video["frames"] for video in video_info]
        )
        assert len(sampler.probabilities) == len(expected_probs)
        # Check that probabilities sum to 1.0
        assert abs(sum(sampler.probabilities) - 1.0) < 1e-10

    def test_init_with_no_files(self, tmp_path: Path):
        """Test initialization with non-existent files."""
        with pytest.raises(ValueError, match="No video files with extensions"):
            RandomVideoFrameSampler(folder=tmp_path / "not_exists", extensions=["mp4"])

    def test_generate_frames(self, video_and_info):
        """Test basic frame generation functionality."""
        video_dir, _ = video_and_info

        # Initialize sampler
        sampler = RandomVideoFrameSampler(folder=video_dir, extensions=["mp4"])

        # Generate multiple frames and test
        for _ in range(10):
            frame = sampler()
            # Check tensor format
            assert isinstance(frame, torch.Tensor)
            assert frame.shape == (3, 64, 64)  # (C, H, W) format
            assert frame.dtype == torch.float32

    def test_max_frames_per_video(self, video_and_info, mocker: MockerFixture):
        """Test max_frames_per_video functionality."""
        video_dir, _ = video_and_info
        max_frames = 3
        # Mock random.choices to control video selection
        # First return index 0 (first video), then index 1 (second video)
        mock_choices = mocker.patch("random.choices")
        mock_choices.side_effect = [
            [0],  # First call returns video index 0
            [1],  # Second call returns video index 1
        ]

        # Mock random.randint to control position within video
        mock_randint = mocker.patch("random.randint")
        mock_randint.return_value = 0  # Always start at the first frame

        # Create sampler with limited frames per video
        sampler = RandomVideoFrameSampler(
            folder=video_dir, extensions=["mp4"], max_frames_per_video=max_frames
        )

        # Get initial frame (should select video 0)
        sampler()
        assert sampler.current_video_index == 0
        assert sampler.frames_read == 1

        # Get remaining frames up to max_frames_per_video
        for _ in range(max_frames - 1):
            sampler()

        assert sampler.frames_read == max_frames

        # Next call should select a new video
        sampler()
        assert sampler.current_video_index == 1  # Should now be video 1
        assert sampler.frames_read == 1  # Counter should be reset

        # Verify mock calls
        assert mock_choices.call_count == 2
        mock_randint.assert_called()

    def test_resource_release(self, video_and_info):
        """Test that resources are properly released."""
        video_dir, _ = video_and_info

        sampler = RandomVideoFrameSampler(folder=video_dir, extensions=["mp4"])
        # Get a frame to initialize VideoCapture object
        sampler()

        # Verify that deleting the sampler doesn't cause exceptions
        # (Resource release happens in __del__)
        sampler.__del__()
        # No exception means success
        assert not sampler.current_video.isOpened()
