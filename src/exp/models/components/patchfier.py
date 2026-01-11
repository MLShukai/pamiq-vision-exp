from typing import override

import torch.nn as nn
from torch import Tensor

from exp.utils import size_3d, size_3d_to_tuple

from ..utils import init_weights


class VideoPatchifier(nn.Module):
    """Convert input videos into spatiotemporal patch (tubelet) embeddings.

    Uses 3D convolution to create non-overlapping tubelets from video
    clips. Default tubelet size is 2x16x16 (temporal x height x width)
    following V-JEPA.
    """

    def __init__(
        self,
        tubelet_size: size_3d = (2, 16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the VideoPatchifier.

        Args:
            tubelet_size: Size of each tubelet as (temporal, height, width).
            in_channels: Number of input video channels.
            embed_dim: Embedding dimension per tubelet.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()
        self._tubelet_size = size_3d_to_tuple(tubelet_size)
        self._embed_dim = embed_dim

        self._proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self._tubelet_size,
            stride=self._tubelet_size,
        )
        init_weights(self._proj, init_std)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Embed video to tubelet patches.

        Args:
            x: Input videos with shape [batch, channels, time, height, width].

        Returns:
            Tubelet embeddings with shape [batch, n_tubelets, embed_dim]
            where n_tubelets = n_t * n_h * n_w.
        """
        # Apply 3D conv: [B, C, T, H, W] -> [B, D, n_t, n_h, n_w]
        x = self._proj(x)

        # Flatten spatial dimensions and transpose
        # [B, D, n_t, n_h, n_w] -> [B, D, n_tubelets] -> [B, n_tubelets, D]
        x = x.flatten(2).transpose(-2, -1)

        return x

    @staticmethod
    def compute_num_tubelets(
        video_shape: tuple[int, int, int], tubelet_size: size_3d
    ) -> tuple[int, int, int]:
        """Compute the number of tubelets in each dimension.

        Args:
            video_shape: Shape of input video as (time, height, width).
            tubelet_size: Size of each tubelet as (temporal, height, width).

        Returns:
            Number of tubelets as (n_temporal, n_height, n_width).

        Raises:
            ValueError: If video dimension is smaller than tubelet size.
        """
        tubelet_size = size_3d_to_tuple(tubelet_size)

        def compute(size: int, tubelet: int, dim_name: str) -> int:
            n = (size - tubelet) // tubelet + 1
            if n <= 0:
                raise ValueError(
                    f"Video {dim_name} {size} is too small for tubelet {dim_name} "
                    f"{tubelet}. Resulting number of tubelets would be {n}."
                )
            return n

        return (
            compute(video_shape[0], tubelet_size[0], "temporal"),
            compute(video_shape[1], tubelet_size[1], "height"),
            compute(video_shape[2], tubelet_size[2], "width"),
        )


class VideoPatchDecoder(nn.Module):
    """Reconstruct videos from tubelet embeddings."""

    def __init__(
        self,
        n_tubelets: tuple[int, int, int],
        tubelet_size: size_3d = (2, 16, 16),
        embed_dim: int = 768,
        out_channels: int = 3,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the VideoPatchDecoder.

        Args:
            n_tubelets: Number of tubelets as (n_temporal, n_height, n_width).
            tubelet_size: Size of each tubelet as (temporal, height, width).
            embed_dim: Dimension of input tubelet embeddings.
            out_channels: Number of output video channels.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()
        self._n_tubelets = n_tubelets
        self._tubelet_size = size_3d_to_tuple(tubelet_size)
        self._embed_dim = embed_dim

        # ConvTranspose3d is the inverse of Conv3d used in VideoPatchifier
        self._unpack = nn.ConvTranspose3d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=self._tubelet_size,
            stride=self._tubelet_size,
        )

        init_weights(self._unpack, init_std)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct videos from tubelet embeddings.

        Args:
            x: Tubelet embeddings with shape [batch_size, n_tubelets, embed_dim]
               where n_tubelets = n_t * n_h * n_w.

        Returns:
            Reconstructed videos with shape [batch_size, out_channels, time, height, width].
        """
        batch_size, n_tubelets, embed_dim = x.shape
        n_t, n_h, n_w = self._n_tubelets
        expected_tubelets = n_t * n_h * n_w

        if n_tubelets != expected_tubelets:
            raise ValueError(
                f"Input has {n_tubelets} tubelets but expected {expected_tubelets}"
            )

        # Reshape: [B, n_tubelets, D] -> [B, D, n_t, n_h, n_w]
        x = x.transpose(-2, -1)  # [B, D, n_tubelets]
        x = x.reshape(batch_size, embed_dim, n_t, n_h, n_w)

        # Apply transpose convolution
        x = self._unpack(x)

        return x
