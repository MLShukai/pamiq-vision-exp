from pathlib import Path

import pytest
import torch
from torch import Tensor

from exp.data.loader import VideoFrameLoader, _center_crop, _preprocess_frame


def _make_mock_video(
    n_frames: int = 30, h: int = 240, w: int = 320, fps: float = 30.0
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Create mock video data matching torchvision.io.read_video output."""
    video = torch.randint(0, 256, (n_frames, h, w, 3), dtype=torch.uint8)
    audio = torch.empty(0)
    info = {"video_fps": fps}
    return video, audio, info


def _write_video_list(tmp_path: Path, paths: list[str | Path]) -> Path:
    """Create a video list file and return its path."""
    list_file = tmp_path / "videos.txt"
    list_file.write_text("\n".join(str(p) for p in paths))
    return list_file


class TestCenterCrop:
    def test_wider_frame_crops_width(self):
        frame = torch.randn(3, 100, 200)
        result = _center_crop(frame, 100, 100)
        assert result.shape == (3, 100, 100)

    def test_taller_frame_crops_height(self):
        frame = torch.randn(3, 200, 100)
        result = _center_crop(frame, 100, 100)
        assert result.shape == (3, 100, 100)

    def test_matching_aspect_ratio_no_crop(self):
        frame = torch.randn(3, 100, 200)
        result = _center_crop(frame, 50, 100)
        assert result.shape == (3, 100, 200)

    def test_crop_is_centered(self):
        frame = torch.zeros(1, 100, 200)
        frame[:, :, 50:150] = 1.0
        result = _center_crop(frame, 100, 100)
        assert result.shape == (1, 100, 100)
        assert result.mean() == 1.0


class TestPreprocessFrame:
    def test_output_shape(self):
        frame = torch.randint(0, 256, (240, 320, 3), dtype=torch.uint8)
        result = _preprocess_frame(frame, (224, 224), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert result.shape == (3, 224, 224)

    def test_output_dtype(self):
        frame = torch.randint(0, 256, (240, 320, 3), dtype=torch.uint8)
        result = _preprocess_frame(frame, (112, 112), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert result.dtype == torch.float32

    def test_normalization(self):
        frame = torch.full((100, 100, 3), 128, dtype=torch.uint8)
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        result = _preprocess_frame(frame, (100, 100), mean, std)
        expected_val = 128.0 / 255.0
        assert torch.allclose(result, torch.full_like(result, expected_val), atol=1e-3)


class TestVideoFrameLoaderValidation:
    def test_empty_paths_raises(self, tmp_path):
        list_file = tmp_path / "empty.txt"
        list_file.write_text("")
        with pytest.raises(ValueError, match="Video list file contains no video paths"):
            VideoFrameLoader(video_list_path=list_file)

    def test_nonexistent_file_raises(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError, match="Video list file not found"):
            VideoFrameLoader(video_list_path=nonexistent)

    def test_negative_fps_raises(self, tmp_path):
        list_file = _write_video_list(tmp_path, ["a.mp4"])
        with pytest.raises(ValueError, match="target_fps must be positive"):
            VideoFrameLoader(video_list_path=list_file, target_fps=-1.0)

    def test_zero_fps_raises(self, tmp_path):
        list_file = _write_video_list(tmp_path, ["a.mp4"])
        with pytest.raises(ValueError, match="target_fps must be positive"):
            VideoFrameLoader(video_list_path=list_file, target_fps=0.0)

    def test_negative_fade_duration_raises(self, tmp_path):
        list_file = _write_video_list(tmp_path, ["a.mp4"])
        with pytest.raises(ValueError, match="fade_duration must be non-negative"):
            VideoFrameLoader(video_list_path=list_file, fade_duration=-1.0)

    def test_non_positive_target_size_raises(self, tmp_path):
        list_file = _write_video_list(tmp_path, ["a.mp4"])
        with pytest.raises(ValueError, match="target_size dimensions must be positive"):
            VideoFrameLoader(video_list_path=list_file, target_size=(0, 224))

    def test_negative_target_size_raises(self, tmp_path):
        list_file = _write_video_list(tmp_path, ["a.mp4"])
        with pytest.raises(ValueError, match="target_size dimensions must be positive"):
            VideoFrameLoader(video_list_path=list_file, target_size=(224, -1))


class TestVideoFrameLoaderSubsampling:
    def test_30fps_to_10fps_skips_frames(self, mocker, tmp_path):
        n_frames = 30
        mock_video = _make_mock_video(n_frames=n_frames, fps=30.0)
        mocker.patch(
            "exp.data.loader.torchvision.io.read_video", return_value=mock_video
        )

        list_file = _write_video_list(tmp_path, ["video.mp4"])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        # skip = round(30/10) = 3, so 30/3 = 10 frames
        assert len(frames) == 10

    def test_same_fps_no_skipping(self, mocker, tmp_path):
        n_frames = 15
        mock_video = _make_mock_video(n_frames=n_frames, fps=15.0)
        mocker.patch(
            "exp.data.loader.torchvision.io.read_video", return_value=mock_video
        )

        list_file = _write_video_list(tmp_path, ["video.mp4"])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=15.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        assert len(frames) == n_frames


class TestVideoFrameLoaderFadeTransition:
    def test_fade_frame_count(self, mocker, tmp_path):
        n_frames = 10
        mock_video = _make_mock_video(n_frames=n_frames, fps=10.0)
        mocker.patch(
            "exp.data.loader.torchvision.io.read_video", return_value=mock_video
        )

        fade_duration = 1.0
        target_fps = 10.0
        fade_frames = int(fade_duration * target_fps)

        list_file = _write_video_list(tmp_path, ["a.mp4", "b.mp4"])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=target_fps,
            fade_duration=fade_duration,
        )
        frames = list(loader)
        # 10 frames per video + fade_out (10) + fade_in (10) = 40
        expected = n_frames * 2 + fade_frames * 2
        assert len(frames) == expected

    def test_no_fade_with_zero_duration(self, mocker, tmp_path):
        n_frames = 10
        mock_video = _make_mock_video(n_frames=n_frames, fps=10.0)
        mocker.patch(
            "exp.data.loader.torchvision.io.read_video", return_value=mock_video
        )

        list_file = _write_video_list(tmp_path, ["a.mp4", "b.mp4"])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        assert len(frames) == n_frames * 2

    def test_fade_out_interpolation(self, mocker, tmp_path):
        # Use a uniform frame so we can verify interpolation values
        video = torch.full((5, 100, 100, 3), 255, dtype=torch.uint8)
        audio = torch.empty(0)
        info = {"video_fps": 5.0}
        mocker.patch(
            "exp.data.loader.torchvision.io.read_video",
            side_effect=[(video, audio, info), (video, audio, info)],
        )

        list_file = _write_video_list(tmp_path, ["a.mp4", "b.mp4"])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=5.0,
            target_size=(100, 100),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            fade_duration=1.0,
        )
        frames = list(loader)
        # 5 frames + 5 fade_out + 5 fade_in + 5 frames = 20
        assert len(frames) == 20

        # Fade-out frames are indices 5..9
        # The last real frame (index 4) has value 1.0 (255/255)
        # Fade out frame 0: alpha = 1 - 1/5 = 0.8
        fade_out_first = frames[5]
        expected_val = 1.0 * (1.0 - 1.0 / 5.0)
        assert torch.allclose(
            fade_out_first, torch.full_like(fade_out_first, expected_val), atol=1e-3
        )

        # Fade out last frame (index 9): alpha = 1 - 5/5 = 0.0 (black)
        fade_out_last = frames[9]
        assert torch.allclose(fade_out_last, torch.zeros_like(fade_out_last), atol=1e-3)


class TestVideoFrameLoaderIterator:
    def test_output_shape(self, mocker, tmp_path):
        mock_video = _make_mock_video(n_frames=10, h=240, w=320, fps=10.0)
        mocker.patch(
            "exp.data.loader.torchvision.io.read_video", return_value=mock_video
        )

        list_file = _write_video_list(tmp_path, ["v.mp4"])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            target_size=(112, 112),
            fade_duration=0.0,
        )
        frames = list(loader)
        assert all(f.shape == (3, 112, 112) for f in frames)

    def test_single_video_total_frames(self, mocker, tmp_path):
        mock_video = _make_mock_video(n_frames=60, fps=30.0)
        mocker.patch(
            "exp.data.loader.torchvision.io.read_video", return_value=mock_video
        )

        list_file = _write_video_list(tmp_path, ["v.mp4"])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        # skip=3, 60/3=20
        assert len(frames) == 20
