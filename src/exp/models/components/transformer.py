import math
from collections.abc import Callable
from functools import partial
from typing import override

import torch.nn as nn
from torch import Tensor

from ..utils import init_weights, rescale_weight_for_depth
from .rope import RoPE3D


class MLP(nn.Module):
    """Multi Layer Perceptron with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the MLP module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features.
            out_features: Number of output features.
            dropout: Dropout probability.
        """
        super().__init__()
        self._fc1 = nn.Linear(in_features, hidden_features)
        self._act = nn.GELU()
        self._fc2 = nn.Linear(hidden_features, out_features)
        self._dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the MLP.
        """
        x = self._fc1(x)
        x = self._act(x)
        x = self._dropout(x)
        x = self._fc2(x)
        x = self._dropout(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with 3D Rotary Position Embedding."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to add bias to the QKV projection.
            attn_drop: Attention dropout probability.
            proj_drop: Output projection dropout probability.
        """
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._scale = 1 / math.sqrt(self._head_dim)

        self._qkv: Callable[[Tensor], Tensor] = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self._attn_drop: Callable[[Tensor], Tensor] = nn.Dropout(attn_drop)
        self._proj = nn.Linear(dim, dim)
        self._proj_drop = nn.Dropout(proj_drop)

    @override
    def forward(self, x: Tensor, rope: RoPE3D) -> Tensor:
        """Forward pass with 3D RoPE.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim].
            rope: 3D RoPE module.

        Returns:
            Output tensor of shape [batch_size, seq_len, dim].
        """
        B, N, C = x.shape

        qkv = self._qkv(x).reshape(B, N, 3, self._num_heads, self._head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Apply 3D RoPE
        q, k = rope(q, k)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self._scale
        attn = attn.softmax(dim=-1)
        attn = self._attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self._proj(x)
        x = self._proj_drop(x)

        return x


class TransformerLayer(nn.Module):
    """Transformer layer with 3D RoPE attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        """Initialize the TransformerLayer.

        Args:
            embedding_dim: Dimension of the embeddings.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
            qkv_bias: Whether to add bias to the QKV projection.
            dropout: Dropout probability for MLP and projection.
            attn_drop: Dropout probability for attention weights.
        """
        super().__init__()
        self._norm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self._attn = Attention(
            embedding_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )
        self._norm2 = nn.LayerNorm(embedding_dim, eps=1e-6)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self._mlp = MLP(
            in_features=embedding_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embedding_dim,
            dropout=dropout,
        )

    @override
    def forward(self, x: Tensor, rope: RoPE3D) -> Tensor:
        """Apply Transformer layer with RoPE.

        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim].
            rope: 3D RoPE module.

        Returns:
            Output tensor with same shape as input.
        """
        x = x + self._attn(self._norm1(x), rope)
        x = x + self._mlp(self._norm2(x))
        return x


class Transformer(nn.Module):
    """Transformer Encoder with 3D RoPE for video processing."""

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        n_temporal: int,
        n_height: int,
        n_width: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        init_std: float = 0.02,
        rope_theta: float = 10000.0,
    ) -> None:
        """Initialize the Transformer.

        Args:
            embedding_dim: Dimension of the embeddings.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            n_temporal: Number of temporal positions (tubelets).
            n_height: Number of height positions (tubelets).
            n_width: Number of width positions (tubelets).
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
            qkv_bias: Whether to add bias to the QKV projection.
            dropout: Dropout probability for MLP and projection.
            attn_drop: Dropout probability for attention weights.
            init_std: Standard deviation for weight initialization.
            rope_theta: Base for RoPE frequency computation.
        """
        super().__init__()
        head_dim = embedding_dim // num_heads

        # Initialize 3D RoPE
        self._rope = RoPE3D(
            dim=head_dim,
            n_temporal=n_temporal,
            n_height=n_height,
            n_width=n_width,
            theta=rope_theta,
        )

        # Build transformer layers
        self._blocks = nn.ModuleList()
        for i in range(depth):
            layer = TransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_drop=attn_drop,
            )
            layer.apply(partial(init_weights, init_std=init_std))
            layer.apply(partial(rescale_weight_for_depth, depth=i + 1))
            self._blocks.append(layer)

        # Final normalization layer
        self._norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        init_weights(self._norm, init_std)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Process input embeddings through transformer layers with RoPE.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            Output tensor of shape [batch_size, seq_len, embedding_dim].
        """
        for block in self._blocks:
            x = block(x, self._rope)

        x = self._norm(x)
        return x
