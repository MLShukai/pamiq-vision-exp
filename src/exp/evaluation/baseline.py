import math
from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from exp.models.components.patchfier import VideoPatchifier
from exp.utils import size_3d, size_3d_to_tuple


class DownsamplingBaseline(nn.Module):
    """Baseline encoder that uses simple spatial downsampling.

    Compresses video [B, C, T, H, W] to a flat feature vector [B,
    feature_size] without learnable parameters. Useful as a
    deterministic lower-bound baseline for encoder evaluation.
    """

    def __init__(
        self,
        feature_size: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self._feature_size = feature_size
        self._in_channels = in_channels
        # Spatial resolution that yields ~feature_size elements after flatten
        pixels_per_channel = feature_size // in_channels
        self._target_spatial = max(round(pixels_per_channel**0.5), 1)

    @override
    def forward(self, video: Tensor) -> Tensor:
        """Encode video by spatial downsampling.

        Args:
            video: Input video [B, C, T, H, W].

        Returns:
            Features [B, feature_size].
        """
        frames = video.mean(dim=2)  # [B, C, H, W] — average over time

        downsampled = F.interpolate(
            frames,
            size=(self._target_spatial, self._target_spatial),
            mode="bilinear",
            align_corners=False,
        )

        flat = downsampled.flatten(1)  # [B, C * spatial^2]
        return self._pad_or_truncate(flat)

    def _pad_or_truncate(self, flat: Tensor) -> Tensor:
        """Adjust feature dimension to exactly ``_feature_size``."""
        n = flat.shape[1]
        if n < self._feature_size:
            return F.pad(flat, (0, self._feature_size - n))
        if n > self._feature_size:
            return flat[:, : self._feature_size]
        return flat


def create_downsampling_baseline(
    video_shape: tuple[int, int, int] = (16, 224, 224),
    tubelet_size: size_3d = (2, 16, 16),
    embed_dim: int = 128,
    in_channels: int = 3,
) -> DownsamplingBaseline:
    """Create a :class:`DownsamplingBaseline` from config parameters.

    Computes ``feature_size = n_tubelets_total * embed_dim`` from
    *video_shape* and *tubelet_size*, making this compatible with
    Hydra-style configs that mirror the encoder's parameters.

    Args:
        video_shape: Video dimensions as (time, height, width).
        tubelet_size: Tubelet size as (temporal, height, width) or scalar.
        embed_dim: Embedding dimension per tubelet.
        in_channels: Number of input video channels.

    Returns:
        Configured :class:`DownsamplingBaseline` instance.
    """
    tubelet_size = size_3d_to_tuple(tubelet_size)
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)
    feature_size = math.prod(n_tubelets) * embed_dim
    return DownsamplingBaseline(feature_size=feature_size, in_channels=in_channels)
