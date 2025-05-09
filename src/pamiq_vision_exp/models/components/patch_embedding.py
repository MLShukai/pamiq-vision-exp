from typing import override

import torch
import torch.nn as nn

from ..utils import init_weights, size_2d


class PatchEmbedding(nn.Module):
    """Convert input images into patch embeddings."""

    @override
    def __init__(
        self,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        init_std: float = 0.02,
    ) -> None:
        """Initializes the PatchEmbedding.

        Args:
            patch_size: Pixel size per a patch.
            in_channels: Num of input images channels.
            embed_dim: Num of embed dimensions per a patch
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        init_weights(self.proj, init_std)

    # (batch, channels, height, width) -> (batch, n_patches, embed_dim)
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(-2).transpose(-2, -1)
        return x
