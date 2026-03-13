from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from exp.models.components.patchfier import VideoPatchifier
from exp.utils import size_3d


class DownsamplingBaseline(nn.Module):
    """Baseline encoder that uses simple spatial downsampling.

    Matches the encoder interface: input [B, C, T, H, W] -> output [B, n_tubelets_total, embed_dim].
    No learnable parameters; provides a deterministic compression baseline.
    """

    def __init__(
        self,
        video_shape: tuple[int, int, int],
        tubelet_size: size_3d,
        embed_dim: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)
        self._n_tubelets_total = n_tubelets[0] * n_tubelets[1] * n_tubelets[2]
        self._embed_dim = embed_dim
        self._in_channels = in_channels

        target_feature_size = self._n_tubelets_total * embed_dim
        pixels_per_channel = target_feature_size // in_channels
        self._target_spatial = max(round(pixels_per_channel**0.5), 1)

    @override
    def forward(self, video: Tensor, masks: Tensor | None = None) -> Tensor:
        """Encode video by spatial downsampling.

        Args:
            video: Input video [B, C, T, H, W].
            masks: Ignored (accepted for interface compatibility).

        Returns:
            Features [B, n_tubelets_total, embed_dim].
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

        # Flatten and reshape to match encoder output
        flat = downsampled.flatten(1)  # [B, C * spatial^2]
        target_size = self._n_tubelets_total * self._embed_dim

        # Pad or truncate to exact target size
        if flat.shape[1] < target_size:
            flat = F.pad(flat, (0, target_size - flat.shape[1]))
        elif flat.shape[1] > target_size:
            flat = flat[:, :target_size]

        return flat.reshape(-1, self._n_tubelets_total, self._embed_dim)
