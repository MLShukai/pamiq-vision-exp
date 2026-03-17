from pathlib import Path

import pytest
import torch
from torch import Tensor

from exp.data.loader import VideoFrameLoader, _center_crop, _preprocess_frame
from tests.helpers import create_test_video


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
        result = _preprocess_frame(frame, (224, 224), True)
        assert result.shape == (3, 224, 224)

    def test_output_dtype(self):
        frame = torch.randint(0, 256, (240, 320, 3), dtype=torch.uint8)
        result = _preprocess_frame(frame, (112, 112), False)
        assert result.dtype == torch.float32

    def test_standardize_uniform_frame(self):
        frame = torch.full((100, 100, 3), 128, dtype=torch.uint8)
        result = _preprocess_frame(frame, (100, 100), True)
        # Uniform input: all values equal, so (val - mean) / (std + eps) ≈ 0
        # Bilinear resize may introduce tiny edge artifacts, so use relaxed atol
        assert torch.allclose(result, torch.zeros_like(result), atol=0.1)

    def test_no_standardize(self):
        frame = torch.full((100, 100, 3), 128, dtype=torch.uint8)
        result = _preprocess_frame(frame, (100, 100), False)
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
    def test_30fps_to_10fps_skips_frames(self, tmp_path):
        n_frames = 30
        video = tmp_path / "video.avi"
        create_test_video(video, n_frames=n_frames, fps=30.0)

        list_file = _write_video_list(tmp_path, [video])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        # skip = round(30/10) = 3, so 30/3 = 10 frames
        assert len(frames) == 10

    def test_same_fps_no_skipping(self, tmp_path):
        n_frames = 15
        video = tmp_path / "video.avi"
        create_test_video(video, n_frames=n_frames, fps=15.0)

        list_file = _write_video_list(tmp_path, [video])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=15.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        assert len(frames) == n_frames


class TestVideoFrameLoaderFadeTransition:
    def test_fade_frame_count(self, tmp_path):
        n_frames = 10
        a = tmp_path / "a.avi"
        b = tmp_path / "b.avi"
        create_test_video(a, n_frames=n_frames, fps=10.0)
        create_test_video(b, n_frames=n_frames, fps=10.0)

        fade_duration = 1.0
        target_fps = 10.0
        fade_frames = int(fade_duration * target_fps)

        list_file = _write_video_list(tmp_path, [a, b])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=target_fps,
            fade_duration=fade_duration,
        )
        frames = list(loader)
        # 10 frames per video + fade_out (10) + fade_in (10) = 40
        expected = n_frames * 2 + fade_frames * 2
        assert len(frames) == expected

    def test_no_fade_with_zero_duration(self, tmp_path):
        n_frames = 10
        a = tmp_path / "a.avi"
        b = tmp_path / "b.avi"
        create_test_video(a, n_frames=n_frames, fps=10.0)
        create_test_video(b, n_frames=n_frames, fps=10.0)

        list_file = _write_video_list(tmp_path, [a, b])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        assert len(frames) == n_frames * 2

    def test_fade_out_interpolation(self, tmp_path):
        # Use uniform frames so we can verify interpolation values
        a = tmp_path / "a.avi"
        b = tmp_path / "b.avi"
        create_test_video(
            a, n_frames=5, height=100, width=100, fps=5.0, pixel_value=255
        )
        create_test_video(
            b, n_frames=5, height=100, width=100, fps=5.0, pixel_value=255
        )

        list_file = _write_video_list(tmp_path, [a, b])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=5.0,
            target_size=(100, 100),
            standardize=False,
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
    def test_output_shape(self, tmp_path):
        video = tmp_path / "v.avi"
        create_test_video(video, n_frames=10, height=240, width=320, fps=10.0)

        list_file = _write_video_list(tmp_path, [video])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            target_size=(112, 112),
            fade_duration=0.0,
        )
        frames = list(loader)
        assert all(f.shape == (3, 112, 112) for f in frames)

    def test_single_video_total_frames(self, tmp_path):
        video = tmp_path / "v.avi"
        create_test_video(video, n_frames=60, fps=30.0)

        list_file = _write_video_list(tmp_path, [video])
        loader = VideoFrameLoader(
            video_list_path=list_file,
            target_fps=10.0,
            fade_duration=0.0,
        )
        frames = list(loader)
        # skip=3, 60/3=20
        assert len(frames) == 20
