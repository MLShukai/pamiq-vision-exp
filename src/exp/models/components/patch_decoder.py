from typing import override

import torch
import torch.nn as nn

from exp.utils import size_2d, size_2d_to_int_tuple

from ..utils import init_weights


class PatchDecoder(nn.Module):
    """Reconstruct images from patch embeddings, performing the inverse
    operation of PatchEmbedding."""

    def __init__(
        self,
        n_patches: size_2d,
        patch_size: size_2d = 16,
        embed_dim: int = 768,
        out_channels: int = 3,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the PatchDecoder.

        Args:
            n_patches: Number of patches along vertical and horizontal axes.
            patch_size: Size of each patch.
            embed_dim: Dimension of input patch embeddings.
            out_channels: Number of output image channels.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()

        self.n_patches = size_2d_to_int_tuple(n_patches)
        self.patch_size = size_2d_to_int_tuple(patch_size)
        self.embed_dim = embed_dim

        # ConvTranspose2d is the direct inverse of Conv2d used in PatchEmbedding
        self.unpatch = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        init_weights(self.unpatch, init_std)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from patch embeddings.

        Args:
            x: Patch embeddings with shape [batch_size, n_patches_h*n_patches_w, embed_dim]

        Returns:
            Reconstructed images with shape [batch_size, out_channels, height, width]
        """
        batch_size, n_patches, embed_dim = x.shape
        n_patches_h, n_patches_w = self.n_patches

        if n_patches != n_patches_h * n_patches_w:
            raise ValueError(
                f"Input has {n_patches} patches but expected {n_patches_h * n_patches_w}"
            )

        # Inverse of transpose(-2, -1) operation in PatchEmbedding
        x = x.transpose(-2, -1)  # [batch_size, embed_dim, n_patches]

        # Inverse of flatten(-2) operation in PatchEmbedding
        x = x.reshape(batch_size, embed_dim, n_patches_h, n_patches_w)

        # Apply transpose convolution - inverse of the conv2d operation
        x = self.unpatch(x)

        return x
