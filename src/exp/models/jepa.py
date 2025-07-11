"""JEPA model components.

This module provides the components for the Joint Embedding Predictive
Architecture (JEPA) model.
"""

import copy
from collections.abc import Callable
from typing import Self, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from exp.utils import size_2d, size_2d_to_int_tuple

from .components.patch_decoder import PatchDecoder
from .components.transformer import Transformer
from .utils import init_weights


class Encoder(nn.Module):
    """Encoder for Joint Embedding Predictive Architecture (JEPA) with mask
    support."""

    def __init__(
        self,
        patchifier: Callable[[Tensor], Tensor] | None = None,
        positional_encodings: Tensor | None = None,
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
            patchifier: Patchfy input data to patch sequence.
            positional_encodings: Positional encoding tensors to be added to patchfied input data.
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

        if positional_encodings is not None:
            if positional_encodings.ndim != 2:
                raise ValueError("positional_encodings must be 2d tensor!")
            if positional_encodings.size(1) != hidden_dim:
                raise ValueError(
                    "positional_encodings channel dimension must be hidden_dim."
                )

        self.patchfier = patchifier if patchifier is not None else nn.Identity()

        # define mask token_vector
        self.mask_token_vector = nn.Parameter(torch.empty(hidden_dim))

        self.positional_encodings: Tensor | None
        self.register_buffer("positional_encodings", positional_encodings)

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
    def forward(self, data: Tensor, masks: Tensor | None = None) -> Tensor:
        """Encode input images into latents, applying masks if provided.

        Args:
            data: Input data
            masks: Boolean masks for images embedded as patches with shape
                [batch_size, n_patches]. True values indicate masked patches.

        Returns:
            Encoded latents with shape [batch_size, n_patches, out_dim]
        """
        # Patchify input images

        x = self.patchfier(data)
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
        if self.positional_encodings is not None:
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

    def clone(self) -> Self:
        """Clone model for creating target or context encoder."""
        return copy.deepcopy(self)


class Predictor(nn.Module):
    """Predictor for Joint Embedding Predictive Architecture (JEPA) with target
    support."""

    def __init__(
        self,
        positional_encodings: Tensor | None = None,
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
        if positional_encodings is not None:
            if positional_encodings.ndim != 2:
                raise ValueError("positional_encodings must be 2d tensor!")
            if positional_encodings.size(1) != hidden_dim:
                raise ValueError(
                    "positional_encodings channel dimension must be hidden_dim."
                )
        self.input_proj = nn.Linear(embed_dim, hidden_dim, bias=True)

        # prepare tokens representing patches to be predicted
        self.prediction_token_vector = nn.Parameter(torch.empty(hidden_dim))

        self.positional_encodings: Tensor | None
        self.register_buffer("positional_encodings", positional_encodings)

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
        latents: Tensor,
        targets: Tensor,
    ) -> Tensor:
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
        if self.positional_encodings is not None:
            x = x + self.positional_encodings

        # Apply transformer
        x = self.transformer(x)

        # Project to output dimension
        x = self.predictor_proj(x)

        return x

    __call__: Callable[[Tensor, Tensor], Tensor]


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
    def forward(self, latents: Tensor) -> Tensor:
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

    __call__: Callable[[Tensor], Tensor]


from pamiq_core.torch import TorchTrainingModel, get_device

from .components.image_patchifier import ImagePatchifier
from .components.positional_embeddings import get_2d_positional_embeddings
from .names import ModelName


def create_image_jepa(
    image_size: size_2d,
    patch_size: size_2d,
    in_channels: int = 3,
    hidden_dim: int = 768,
    embed_dim: int = 128,
    depth: int = 6,
    num_heads: int = 3,
) -> dict[ModelName, nn.Module]:
    """Create a complete Image JEPA (Joint Embedding Predictive Architecture)
    model.

    This factory function creates all components needed for Image JEPA training and inference,
    including context encoder, target encoder, predictor, and inference pooling. The target
    encoder is initialized as a clone of the context encoder for momentum-based updates.

    Args:
        image_size: Input image dimensions as (height, width) or single int for square images.
        patch_size: Patch dimensions as (height, width) or single int for square patches.
        in_channels: Number of input image channels (e.g., 3 for RGB).
        hidden_dim: Hidden dimension for encoder transformers.
        embed_dim: Output embedding dimension for encoders.
        depth: Number of transformer layers in encoders.
        num_heads: Number of attention heads in encoders.
        output_downsample: Downsampling factor for inference pooling as (height, width) or single int.

    Returns:
        A tuple containing:
            - context_encoder: Encoder for processing masked images
            - target_encoder: Encoder clone for generating targets (updated via EMA)
            - predictor: Predictor for reconstructing target patches from context
            - infer: AveragePoolInfer for downsampled inference
            - num_patches: Final patch dimensions after downsampling as (height, width)

    NOTE:
        The predictor uses half the hidden dimensions and attention heads of the encoders
        for efficiency. The target encoder should be updated using exponential moving
        average of the context encoder parameters during training.
    """
    patchifier = ImagePatchifier(
        patch_size,
        in_channels=in_channels,
        embed_dim=hidden_dim,
    )
    num_patches = ImagePatchifier.compute_num_patches(image_size, patch_size)

    context_encoder = Encoder(
        patchifier,
        get_2d_positional_embeddings(hidden_dim, num_patches).reshape(-1, hidden_dim),
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )

    target_encoder = context_encoder.clone()

    predictor = Predictor(
        get_2d_positional_embeddings(hidden_dim // 2, num_patches).reshape(
            -1, hidden_dim // 2
        ),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim // 2,
        depth=depth,
        num_heads=num_heads // 2,
    )

    return {
        ModelName.JEPA_CONTEXT_ENCODER: context_encoder,
        ModelName.JEPA_TARGET_ENCODER: target_encoder,
        ModelName.JEPA_PREDICTOR: predictor,
    }
