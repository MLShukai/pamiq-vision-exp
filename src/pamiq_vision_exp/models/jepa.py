"""JEPA model components.

This module provides the components for the Joint Embedding Predictive
Architecture (JEPA) model.
"""

from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.patch_decoder import PatchDecoder
from .components.patch_embedding import PatchEmbedding
from .components.positional_embeddings import get_2d_positional_embeddings
from .components.transformer import Transformer
from .utils import init_weights, size_2d, size_2d_to_int_tuple


class Encoder(nn.Module):
    """Encoder for Joint Embedding Predictive Architecture (JEPA) with mask
    support."""

    def __init__(
        self,
        img_size: size_2d = 224,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        hidden_dim: int = 768,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the JEPAEncoder.

        Args:
            img_size: Input image size.
            patch_size: Pixel size per patch.
            in_channels: Input image channels.
            hidden_dim: Hidden dimension per patch.
            embed_dim: Output dimension per patch.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            qkv_bias: Whether to use bias in query, key, value projections.
            qk_scale: Scale factor for query-key dot product.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()
        self.num_features = self.embed_dim = hidden_dim
        self.num_heads = num_heads

        # define input layer to convert input image into patches.
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_dim,
        )

        # define mask token_vector
        self.mask_token_vector = nn.Parameter(torch.empty(hidden_dim))

        # define positional encodings
        img_size = size_2d_to_int_tuple(img_size)
        patch_size = size_2d_to_int_tuple(patch_size)
        img_height, img_width = img_size
        patch_height, patch_width = patch_size

        if img_height % patch_height != 0:
            raise ValueError(
                f"Image height {img_height} must be divisible by patch height {patch_height}"
            )
        if img_width % patch_width != 0:
            raise ValueError(
                f"Image width {img_width} must be divisible by patch width {patch_width}"
            )

        n_patches_hw = (img_height // patch_height, img_width // patch_width)
        n_patches = n_patches_hw[0] * n_patches_hw[1]

        positional_encodings = get_2d_positional_embeddings(
            hidden_dim,
            n_patches_hw,
        ).reshape(1, n_patches, hidden_dim)
        self.positional_encodings: torch.Tensor
        self.register_buffer(
            "positional_encodings", torch.from_numpy(positional_encodings).float()
        )

        # define transformer
        self.transformer = Transformer(
            embedding_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=drop_rate,
            attn_drop=attn_drop_rate,
            init_std=init_std,
        )

        self.out_proj = nn.Linear(hidden_dim, embed_dim)

        # initialize
        nn.init.trunc_normal_(self.mask_token_vector, std=init_std)
        init_weights(self.out_proj, init_std)

    @override
    def forward(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode input images into latents, applying masks if provided.

        Args:
            images: Input images with shape [batch_size, 3, height, width]
            masks: Boolean masks for images embedded as patches with shape
                [batch_size, n_patches]. True values indicate masked patches.

        Returns:
            Encoded latents with shape [batch_size, n_patches, out_dim]
        """
        # Patchify input images
        x = self.patch_embed(images)
        # x: [batch_size, n_patches, embed_dim]

        # Apply mask if provided
        if masks is not None:
            if x.shape[:-1] != masks.shape:
                raise ValueError(
                    f"Shape mismatch: x{x.shape[:-1]} vs masks{masks.shape}"
                )
            if masks.dtype != torch.bool:
                raise ValueError(
                    f"Mask tensor dtype must be bool. input: {masks.dtype}"
                )
            x = x.clone()  # Avoid breaking gradient graph
            x[masks] = self.mask_token_vector

        # Add positional embedding to x
        x = x + self.positional_encodings

        # Apply transformer
        x = self.transformer(x)

        # Project to output dimension
        x = self.out_proj(x)
        return x

    @override
    def __call__(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        return super().__call__(images, masks)


class Predictor(nn.Module):
    """Predictor for Joint Embedding Predictive Architecture (JEPA) with target
    support."""

    def __init__(
        self,
        n_patches: size_2d,
        embed_dim: int = 384,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the JEPAPredictor.

        Args:
            n_patches: Number of patches along vertical and horizontal axes.
            embed_dim: Output dimension of the context encoder.
            hidden_dim: Hidden dimension for prediction.
            depth: Number of transformer layers.
            num_heads: Number of attention heads for transformer layers.
            mlp_ratio: Ratio for MLP hidden dimension in transformer layers.
            qkv_bias: Whether to use bias in query, key, value projections.
            qk_scale: Scale factor for query-key dot product.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            init_std: Standard deviation for weight initialization.
        """
        super().__init__()

        self.input_proj = nn.Linear(embed_dim, hidden_dim, bias=True)

        # prepare tokens representing patches to be predicted
        self.prediction_token_vector = nn.Parameter(torch.empty(hidden_dim))

        # define positional encodings
        (n_patches_vertical, n_patches_horizontal) = size_2d_to_int_tuple(n_patches)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, grid_size=(n_patches_vertical, n_patches_horizontal)
        ).reshape(1, n_patches_vertical * n_patches_horizontal, hidden_dim)

        self.positional_encodings: torch.Tensor
        self.register_buffer(
            "positional_encodings", torch.from_numpy(positional_encodings).float()
        )

        # define transformer
        self.transformer = Transformer(
            embedding_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout=drop_rate,
            attn_drop=attn_drop_rate,
            init_std=init_std,
        )

        self.predictor_proj = nn.Linear(hidden_dim, embed_dim, bias=True)

        # initialize
        nn.init.trunc_normal_(self.prediction_token_vector, std=init_std)
        init_weights(self.input_proj, init_std)
        init_weights(self.predictor_proj, init_std)

    @override
    def forward(
        self,
        latents: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Predict latents of target patches based on input latents and boolean
        targets.

        Args:
            latents: Input latents from context_encoder with shape
                [batch, n_patches, embed_dim]
            targets: Boolean targets for patches with shape [batch, n_patches].
                True values indicate target patches to be predicted.

        Returns:
            Prediction results for target patches with shape
                [batch, n_patches, embed_dim]
        """
        # Map from encoder-dim to predictor-dim
        x = self.input_proj(latents)

        # Apply targets: adding prediction tokens
        if x.shape[:-1] != targets.shape:
            raise ValueError(
                f"Shape mismatch: x{x.shape[:-1]} vs targets{targets.shape}"
            )
        if targets.dtype != torch.bool:
            raise ValueError(
                f"Target tensor dtype must be bool. input: {targets.dtype}"
            )

        x = x.clone()  # Avoid breaking gradient graph
        x[targets] += self.prediction_token_vector

        # Add positional encodings
        x = x + self.positional_encodings

        # Apply transformer
        x = self.transformer(x)

        # Project to output dimension
        x = self.predictor_proj(x)

        return x

    @override
    def __call__(self, latents: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return super().__call__(latents, targets)


class LightWeightDecoder(nn.Module):
    """Lightweight decoder for reconstructing images from patch embeddings with
    optional upsampling.

    This module takes patch embeddings and reconstructs them into
    images. It can optionally upsample the patch grid using transposed
    convolution before decoding.
    """

    def __init__(
        self,
        n_patches: size_2d,
        patch_size: size_2d = 16,
        embed_dim: int = 768,
        out_channels: int = 3,
        init_std: float = 0.02,
        upsample: size_2d = 1,
    ) -> None:
        """Initialize the lightweight decoder.

        Args:
            n_patches: Number of patches along vertical and horizontal axes.
            patch_size: Size of each patch.
            embed_dim: Dimension of input patch embeddings.
            out_channels: Number of output image channels.
            init_std: Standard deviation for weight initialization.
            upsample: Factor by which to upsample patches before decoding.

        Raises:
            ValueError: If any dimension of upsample is less than 1.
        """
        super().__init__()
        upsample = size_2d_to_int_tuple(upsample)
        if any(up < 1 for up in upsample):
            raise ValueError("upsample must be larger than 0.")
        n_patches = size_2d_to_int_tuple(n_patches)

        self.n_patches = n_patches
        self.upsample = upsample

        self.upsample_conv = nn.ConvTranspose2d(
            embed_dim, embed_dim, upsample, upsample
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.patch_decoder = PatchDecoder(
            self.upsampled_n_patches, patch_size, embed_dim, out_channels
        )

        # 修正: self.upsampleではなくself.upsample_convに適用
        for layer in [self.upsample_conv, self.proj]:
            init_weights(layer, init_std)

    @property
    def upsampled_n_patches(self) -> tuple[int, int]:
        """Calculate the number of patches after upsampling.

        Returns:
            A tuple containing the number of patches along vertical and horizontal axes after upsampling.
        """
        return self.n_patches[0] * self.upsample[0], self.n_patches[1] * self.upsample[
            1
        ]

    @override
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode input latents into reconstructed image.

        Args:
            latents: Input latents with shape [batch_size, n_patches, embed_dim].

        Returns:
            Reconstructed image with shape [batch_size, out_channels, height, width].
        """
        x = latents.transpose(-1, -2)  # [batch, dim, patch_flatten]
        x = x.reshape(*x.shape[:2], *self.n_patches)  # [batch, dim, patch_v, patch_h]
        x = F.gelu(self.upsample_conv(x))  # [batch, dim, patch_v', patch_h']
        x = x.flatten(-2).transpose(-1, -2)  # [batch, patch_flatten, dim]
        x = F.gelu(self.proj(x))
        x = self.patch_decoder(x)  # [batch, channels, height, width]
        return x

    @override
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return super().__call__(latents)
