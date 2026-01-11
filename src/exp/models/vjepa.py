"""Video JEPA model components.

This module provides the components for Video Joint Embedding Predictive
Architecture (V-JEPA) model following V-JEPA2 design with 3D RoPE.
"""

import copy
from typing import Self, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from exp.utils import size_3d, size_3d_to_tuple

from .components import (
    Transformer,
    VideoPatchDecoder,
    VideoPatchifier,
)
from .utils import init_weights


class Encoder(nn.Module):
    """Encoder for Video JEPA with 3D RoPE and mask support."""

    def __init__(
        self,
        patchifier: VideoPatchifier | None = None,
        n_tubelets: tuple[int, int, int] = (8, 14, 14),
        hidden_dim: int = 768,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
        rope_theta: float = 10000.0,
    ) -> None:
        """Initialize the VideoEncoder.

        Args:
            patchifier: Video patchifier to convert videos to tubelets.
            n_tubelets: Number of tubelets as (n_temporal, n_height, n_width).
            hidden_dim: Hidden dimension per tubelet.
            embed_dim: Output embedding dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio for MLP hidden dimension.
            qkv_bias: Whether to use bias in QKV projections.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            init_std: Standard deviation for weight initialization.
            rope_theta: Base for RoPE frequency computation.
        """
        super().__init__()
        self._hidden_dim = hidden_dim
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._n_tubelets = n_tubelets

        self._patchifier = patchifier if patchifier is not None else nn.Identity()

        # Mask token for masked patches
        self._mask_token = nn.Parameter(torch.empty(hidden_dim))

        # Transformer with 3D RoPE
        n_t, n_h, n_w = n_tubelets
        self._transformer = Transformer(
            embedding_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            n_temporal=n_t,
            n_height=n_h,
            n_width=n_w,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=drop_rate,
            attn_drop=attn_drop_rate,
            init_std=init_std,
            rope_theta=rope_theta,
        )

        self._out_proj = nn.Linear(hidden_dim, embed_dim)

        # Initialize
        nn.init.trunc_normal_(self._mask_token, std=init_std)
        init_weights(self._out_proj, init_std)

    @override
    def forward(self, video: Tensor, masks: Tensor | None = None) -> Tensor:
        """Encode input videos into latents with optional masking.

        Args:
            video: Input video [batch, channels, time, height, width].
            masks: Boolean masks [batch, n_tubelets]. True = masked.

        Returns:
            Encoded latents [batch, n_tubelets, embed_dim].
        """
        x: Tensor = self._patchifier(video)

        if masks is not None:
            if x.shape[:-1] != masks.shape:
                raise ValueError(
                    f"Shape mismatch: x{x.shape[:-1]} vs masks{masks.shape}"
                )
            if masks.dtype != torch.bool:
                raise ValueError(f"Mask dtype must be bool, got {masks.dtype}")
            x = x.clone()
            x[masks] = self._mask_token

        x = self._transformer(x)
        x = self._out_proj(x)

        return x

    def clone(self) -> Self:
        """Clone model for creating target encoder."""
        return copy.deepcopy(self)


class Predictor(nn.Module):
    """Predictor for Video JEPA with 3D RoPE."""

    def __init__(
        self,
        n_tubelets: tuple[int, int, int] = (8, 14, 14),
        embed_dim: int = 384,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_std: float = 0.02,
        rope_theta: float = 10000.0,
    ) -> None:
        """Initialize the VideoPredictor.

        Args:
            n_tubelets: Number of tubelets as (n_temporal, n_height, n_width).
            embed_dim: Input embedding dimension from encoder.
            hidden_dim: Hidden dimension for prediction.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio for MLP hidden dimension.
            qkv_bias: Whether to use bias in QKV projections.
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            init_std: Standard deviation for weight initialization.
            rope_theta: Base for RoPE frequency computation.
        """
        super().__init__()
        self._n_tubelets = n_tubelets

        self._input_proj = nn.Linear(embed_dim, hidden_dim)
        self._prediction_token = nn.Parameter(torch.empty(hidden_dim))

        n_t, n_h, n_w = n_tubelets
        self._transformer = Transformer(
            embedding_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            n_temporal=n_t,
            n_height=n_h,
            n_width=n_w,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=drop_rate,
            attn_drop=attn_drop_rate,
            init_std=init_std,
            rope_theta=rope_theta,
        )

        self._out_proj = nn.Linear(hidden_dim, embed_dim)

        nn.init.trunc_normal_(self._prediction_token, std=init_std)
        init_weights(self._input_proj, init_std)
        init_weights(self._out_proj, init_std)

    @override
    def forward(self, latents: Tensor, targets: Tensor) -> Tensor:
        """Predict latents of target tubelets.

        Args:
            latents: Input latents [batch, n_tubelets, embed_dim].
            targets: Boolean targets [batch, n_tubelets]. True = predict.

        Returns:
            Predictions [batch, n_tubelets, embed_dim].
        """
        x: Tensor = self._input_proj(latents)

        if x.shape[:-1] != targets.shape:
            raise ValueError(
                f"Shape mismatch: x{x.shape[:-1]} vs targets{targets.shape}"
            )
        if targets.dtype != torch.bool:
            raise ValueError(f"Target dtype must be bool, got {targets.dtype}")

        x = x.clone()
        x[targets] += self._prediction_token

        x = self._transformer(x)
        x = self._out_proj(x)

        return x


class LightWeightDecoder(nn.Module):
    """Lightweight decoder for video reconstruction from tubelet embeddings."""

    def __init__(
        self,
        n_tubelets: tuple[int, int, int],
        tubelet_size: size_3d = (2, 16, 16),
        embed_dim: int = 768,
        out_channels: int = 3,
        init_std: float = 0.02,
        upsample: size_3d = 1,
    ) -> None:
        """Initialize the lightweight video decoder.

        Args:
            n_tubelets: Number of tubelets as (n_temporal, n_height, n_width).
            tubelet_size: Size of each tubelet.
            embed_dim: Dimension of input embeddings.
            out_channels: Number of output video channels.
            init_std: Standard deviation for weight initialization.
            upsample: Upsampling factor for each dimension.
        """
        super().__init__()
        upsample = size_3d_to_tuple(upsample)
        if any(up < 1 for up in upsample):
            raise ValueError("upsample must be >= 1 in all dimensions.")

        self._n_tubelets = n_tubelets
        self._upsample = upsample

        self._upsample_conv = nn.ConvTranspose3d(
            embed_dim, embed_dim, upsample, upsample
        )
        self._proj = nn.Linear(embed_dim, embed_dim)

        upsampled_n_tubelets = (
            n_tubelets[0] * upsample[0],
            n_tubelets[1] * upsample[1],
            n_tubelets[2] * upsample[2],
        )

        self._video_decoder = VideoPatchDecoder(
            upsampled_n_tubelets, tubelet_size, embed_dim, out_channels
        )

        for layer in [self._upsample_conv, self._proj]:
            init_weights(layer, init_std)

    @property
    def upsampled_n_tubelets(self) -> tuple[int, int, int]:
        """Get number of tubelets after upsampling."""
        return (
            self._n_tubelets[0] * self._upsample[0],
            self._n_tubelets[1] * self._upsample[1],
            self._n_tubelets[2] * self._upsample[2],
        )

    @override
    def forward(self, latents: Tensor) -> Tensor:
        """Decode latents to video.

        Args:
            latents: Input latents [batch, n_tubelets, embed_dim].

        Returns:
            Reconstructed video [batch, channels, time, height, width].
        """
        n_t, n_h, n_w = self._n_tubelets

        x = latents.transpose(-1, -2)
        x = x.reshape(*x.shape[:2], n_t, n_h, n_w)
        x = F.gelu(self._upsample_conv(x))
        x = x.flatten(2).transpose(-1, -2)
        x = F.gelu(self._proj(x))
        x = self._video_decoder(x)

        return x


def create_video_jepa(
    video_shape: tuple[int, int, int],
    tubelet_size: size_3d = (2, 16, 16),
    in_channels: int = 3,
    hidden_dim: int = 768,
    embed_dim: int = 128,
    depth: int = 6,
    num_heads: int = 12,
    predictor_hidden_dim: int | None = None,
    predictor_depth: int | None = None,
    predictor_num_heads: int | None = None,
) -> dict[str, nn.Module]:
    """Create a complete Video JEPA model.

    Args:
        video_shape: Input video shape as (time, height, width).
        tubelet_size: Tubelet size as (temporal, height, width).
        in_channels: Number of input video channels.
        hidden_dim: Hidden dimension for encoder transformers.
        embed_dim: Output embedding dimension.
        depth: Number of transformer layers in encoders.
        num_heads: Number of attention heads in encoders.
        predictor_hidden_dim: Hidden dim for predictor (default: hidden_dim // 2).
        predictor_depth: Depth for predictor (default: depth).
        predictor_num_heads: Heads for predictor (default: num_heads // 2).

    Returns:
        Dictionary with 'context_encoder', 'target_encoder', and 'predictor'.
    """
    tubelet_size = size_3d_to_tuple(tubelet_size)
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)

    patchifier = VideoPatchifier(
        tubelet_size=tubelet_size,
        in_channels=in_channels,
        embed_dim=hidden_dim,
    )

    context_encoder = Encoder(
        patchifier=patchifier,
        n_tubelets=n_tubelets,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )

    target_encoder = context_encoder.clone()

    pred_hidden = predictor_hidden_dim or hidden_dim // 2
    pred_depth = predictor_depth or depth
    pred_heads = predictor_num_heads or max(num_heads // 2, 1)

    predictor = Predictor(
        n_tubelets=n_tubelets,
        embed_dim=embed_dim,
        hidden_dim=pred_hidden,
        depth=pred_depth,
        num_heads=pred_heads,
    )

    return {
        "context_encoder": context_encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
    }
