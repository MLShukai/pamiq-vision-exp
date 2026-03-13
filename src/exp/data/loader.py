"""Video frame loader with preprocessing and fade transitions."""

from collections.abc import Iterator
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from torch import Tensor


def _get_video_fps(path: Path) -> float:
    """Get the average frame rate of a video file.

    Args:
        path: Path to the video file.

    Returns:
        Average FPS as a float.
    """
    container = av.open(str(path))
    rate = container.streams.video[0].average_rate
    container.close()
    if rate is None:
        raise ValueError(f"Cannot determine FPS for video: {path}")
    return float(rate)


def _iter_video_frames(path: Path, skip: int) -> Iterator[Tensor]:
    """Iterate over video frames, yielding every `skip`-th frame as a tensor.

    Args:
        path: Path to the video file.
        skip: Yield one frame every `skip` frames.

    Yields:
        Frame tensor [H, W, C] uint8.
    """
    container = av.open(str(path))
    for frame_idx, frame in enumerate(container.decode(video=0)):
        if frame_idx % skip == 0:
            yield torch.from_numpy(frame.to_ndarray(format="rgb24"))
    container.close()


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
    standardize: bool,
) -> Tensor:
    """Preprocess a single video frame.

    Pipeline: convert to float [0,1] -> center crop -> resize -> standardize.

    Args:
        frame: Input frame [H, W, C] uint8.
        target_size: Target (H, W).
        standardize: If True, normalize to zero mean and unit variance.

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

    # Standardize
    if standardize:
        mean = out.mean()
        std = out.std()
        out = (out - mean) / (std + 1e-6)

    return out


class VideoFrameLoader:
    """Loads video frames sequentially with preprocessing and fade
    transitions."""

    def __init__(
        self,
        video_list_path: str | Path,
        target_fps: float = 10.0,
        target_size: tuple[int, int] = (224, 224),
        standardize: bool = True,
        fade_duration: float = 1.0,
    ) -> None:
        """Initialize the video frame loader.

        Args:
            video_list_path: Path to a text file listing video paths, one per
                line. Blank lines are ignored.
            target_fps: Target frames per second for subsampling.
            target_size: Target frame size as (H, W).
            standardize: If True, normalize frames to zero mean and unit
                variance.
            fade_duration: Duration of fade transitions in seconds.

        Raises:
            FileNotFoundError: If video_list_path does not exist.
            ValueError: If the video list file contains no paths,
                target_fps <= 0, fade_duration < 0, or target_size
                dimensions are not positive.
        """
        list_path = Path(video_list_path)
        if not list_path.exists():
            raise FileNotFoundError(f"Video list file not found: {list_path}")

        video_paths = [
            Path(line.strip())
            for line in list_path.read_text().splitlines()
            if line.strip()
        ]

        if len(video_paths) == 0:
            raise ValueError("Video list file contains no video paths.")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive.")
        if fade_duration < 0:
            raise ValueError("fade_duration must be non-negative.")
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(
                f"target_size dimensions must be positive, got {target_size}"
            )

        self._video_paths = video_paths
        self._target_fps = target_fps
        self._target_size = target_size
        self._standardize = standardize
        self._fade_duration = fade_duration

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over preprocessed video frames with fade transitions.

        Yields:
            Preprocessed frames [C, H, W] float32.
        """
        fade_frames = int(self._fade_duration * self._target_fps)
        last_frame: Tensor | None = None

        for video_idx, path in enumerate(self._video_paths):
            fps = _get_video_fps(path)
            skip = max(1, round(fps / self._target_fps))

            frame_iter = _iter_video_frames(path, skip)
            first_raw = next(frame_iter, None)
            if first_raw is None:
                continue

            # Preprocess first frame for potential fade-in
            first_frame = _preprocess_frame(
                first_raw, self._target_size, self._standardize
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

            # Yield first frame
            last_frame = first_frame
            yield first_frame

            # Yield remaining subsampled frames
            for raw_frame in frame_iter:
                processed = _preprocess_frame(
                    raw_frame, self._target_size, self._standardize
                )
                last_frame = processed
                yield processed
