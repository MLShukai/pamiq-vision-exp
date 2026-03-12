"""Video frame loader with preprocessing and fade transitions."""

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.io
from torch import Tensor


def _center_crop(frame: Tensor, target_h: int, target_w: int) -> Tensor:
    """Center crop a frame to match the target aspect ratio.

    Args:
        frame: Input frame [C, H, W].
        target_h: Target height for aspect ratio.
        target_w: Target width for aspect ratio.

    Returns:
        Center-cropped frame [C, crop_h, crop_w].
    """
    _, h, w = frame.shape
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Too wide, crop width
        new_w = int(h * target_ratio)
        start = (w - new_w) // 2
        return frame[:, :, start : start + new_w]
    else:
        # Too tall, crop height
        new_h = int(w / target_ratio)
        start = (h - new_h) // 2
        return frame[:, start : start + new_h, :]


def _preprocess_frame(
    frame: Tensor,
    target_size: tuple[int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Tensor:
    """Preprocess a single video frame.

    Pipeline: convert to float [0,1] -> center crop -> resize -> normalize.

    Args:
        frame: Input frame [H, W, C] uint8.
        target_size: Target (H, W).
        mean: Normalization mean per channel.
        std: Normalization std per channel.

    Returns:
        Preprocessed frame [C, H, W] float32.
    """
    # [H, W, C] -> [C, H, W] and convert to float [0, 1]
    out = frame.permute(2, 0, 1).float() / 255.0

    # Center crop to match target aspect ratio
    target_h, target_w = target_size
    out = _center_crop(out, target_h, target_w)

    # Resize using interpolate (expects [N, C, H, W])
    out = F.interpolate(
        out.unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Normalize
    mean_t = torch.tensor(mean, dtype=out.dtype, device=out.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=out.dtype, device=out.device).view(-1, 1, 1)
    out = (out - mean_t) / std_t

    return out


class VideoFrameLoader:
    """Loads video frames sequentially with preprocessing and fade
    transitions."""

    def __init__(
        self,
        video_paths: list[Path] | list[str],
        target_fps: float = 10.0,
        target_size: tuple[int, int] = (224, 224),
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        fade_duration: float = 1.0,
    ) -> None:
        """Initialize the video frame loader.

        Args:
            video_paths: List of paths to video files.
            target_fps: Target frames per second for subsampling.
            target_size: Target frame size as (H, W).
            mean: Normalization mean per channel.
            std: Normalization std per channel.
            fade_duration: Duration of fade transitions in seconds.

        Raises:
            ValueError: If video_paths is empty, target_fps <= 0,
                fade_duration < 0, or target_size dimensions are not positive.
        """
        if len(video_paths) == 0:
            raise ValueError("video_paths must not be empty.")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive.")
        if fade_duration < 0:
            raise ValueError("fade_duration must be non-negative.")
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(
                f"target_size dimensions must be positive, got {target_size}"
            )

        self._video_paths = [Path(p) for p in video_paths]
        self._target_fps = target_fps
        self._target_size = target_size
        self._mean = mean
        self._std = std
        self._fade_duration = fade_duration

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over preprocessed video frames with fade transitions.

        Yields:
            Preprocessed frames [C, H, W] float32.
        """
        fade_frames = int(self._fade_duration * self._target_fps)
        last_frame: Tensor | None = None

        for video_idx, path in enumerate(self._video_paths):
            video, _audio, info = torchvision.io.read_video(str(path), pts_unit="sec")
            source_fps: float = info["video_fps"]
            skip = max(1, round(source_fps / self._target_fps))

            # Subsample frames
            subsampled = video[::skip]

            if subsampled.shape[0] == 0:
                continue

            # Preprocess first frame for potential fade-in
            first_frame = _preprocess_frame(
                subsampled[0], self._target_size, self._mean, self._std
            )

            # Fade transition between videos
            if video_idx > 0 and fade_frames > 0 and last_frame is not None:
                black = torch.zeros_like(last_frame)
                # Fade out: last_frame -> black
                for i in range(fade_frames):
                    alpha = 1.0 - (i + 1) / fade_frames
                    yield last_frame * alpha + black * (1.0 - alpha)
                # Fade in: black -> first_frame
                for i in range(fade_frames):
                    alpha = (i + 1) / fade_frames
                    yield first_frame * alpha + black * (1.0 - alpha)

            # Yield all subsampled frames
            for frame_idx in range(subsampled.shape[0]):
                processed = _preprocess_frame(
                    subsampled[frame_idx],
                    self._target_size,
                    self._mean,
                    self._std,
                )
                last_frame = processed
                yield processed
