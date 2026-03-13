from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from exp.models.components.patchfier import VideoPatchifier
from exp.utils import size_3d, size_3d_to_tuple


class DownsamplingBaseline(nn.Module):
    """Baseline encoder that uses simple spatial downsampling.

    Input [B, C, T, H, W] -> output [B, feature_size]. No learnable
    parameters; provides a deterministic compression baseline.
    """

    def __init__(
        self,
        feature_size: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self._feature_size = feature_size
        self._in_channels = in_channels
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
        # Average over time dimension
        frames = video.mean(dim=2)  # [B, C, H, W]

        # Downsample spatially
        downsampled = F.interpolate(
            frames,
            size=(self._target_spatial, self._target_spatial),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten and pad/truncate to exact feature size
        flat = downsampled.flatten(1)  # [B, C * spatial^2]

        if flat.shape[1] < self._feature_size:
            flat = F.pad(flat, (0, self._feature_size - flat.shape[1]))
        elif flat.shape[1] > self._feature_size:
            flat = flat[:, : self._feature_size]

        return flat


def create_downsampling_baseline(
    video_shape: tuple[int, int, int] = (16, 224, 224),
    tubelet_size: size_3d = (2, 16, 16),
    embed_dim: int = 128,
    in_channels: int = 3,
) -> DownsamplingBaseline:
    """Create a DownsamplingBaseline from config parameters.

    Accepts video_shape/tubelet_size/embed_dim for Hydra config
    compatibility. Computes feature_size = n_tubelets_total * embed_dim
    internally.
    """
    tubelet_size = size_3d_to_tuple(tubelet_size)
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)
    feature_size = n_tubelets[0] * n_tubelets[1] * n_tubelets[2] * embed_dim
    return DownsamplingBaseline(feature_size=feature_size, in_channels=in_channels)
