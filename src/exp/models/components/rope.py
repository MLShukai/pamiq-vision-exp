from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
from torch import Tensor


def _compute_rope_frequencies(dim: int, seq_len: int, theta: float = 10000.0) -> Tensor:
    """Compute frequency bands for rotary position embedding.

    Args:
        dim: Dimension of the embedding (must be even).
        seq_len: Sequence length.
        theta: Base for frequency computation.

    Returns:
        Complex tensor of shape [seq_len, dim // 2] for rotation.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)  # [seq_len, dim // 2]
    return torch.polar(torch.ones_like(angles), angles)


def _apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embedding to input tensor.

    Args:
        x: Input tensor of shape [..., seq_len, dim].
        freqs: Complex frequency tensor of shape [seq_len, dim // 2].

    Returns:
        Rotated tensor with same shape as input.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs.to(x.device)
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class RoPE3D(nn.Module):
    """3D Rotary Position Embedding for spatiotemporal sequences.

    Partitions the feature dimension into three segments for temporal,
    height, and width axes, applying 1D rotary embeddings to each.
    """

    def __init__(
        self,
        dim: int,
        n_temporal: int,
        n_height: int,
        n_width: int,
        theta: float = 10000.0,
    ) -> None:
        """Initialize RoPE3D.

        Args:
            dim: Head dimension (embedding_dim // num_heads).
            n_temporal: Number of temporal positions.
            n_height: Number of height positions.
            n_width: Number of width positions.
            theta: Base for frequency computation.
        """
        super().__init__()
        self._dim = dim
        self._n_temporal = n_temporal
        self._n_height = n_height
        self._n_width = n_width

        # Partition dimension into three segments (ensure even for complex ops)
        self._dim_t = (dim // 3) // 2 * 2
        self._dim_h = (dim // 3) // 2 * 2
        self._dim_w = dim - self._dim_t - self._dim_h
        if self._dim_w % 2 != 0:
            self._dim_w -= 1
            self._dim_h += 1

        # Precompute frequencies for each axis
        self.register_buffer(
            "_freqs_t", _compute_rope_frequencies(self._dim_t, n_temporal, theta)
        )
        self.register_buffer(
            "_freqs_h", _compute_rope_frequencies(self._dim_h, n_height, theta)
        )
        self.register_buffer(
            "_freqs_w", _compute_rope_frequencies(self._dim_w, n_width, theta)
        )

        # Precompute position indices
        t_pos = torch.arange(n_temporal)
        h_pos = torch.arange(n_height)
        w_pos = torch.arange(n_width)
        grid_t, grid_h, grid_w = torch.meshgrid(t_pos, h_pos, w_pos, indexing="ij")
        self.register_buffer("_positions_t", grid_t.flatten())
        self.register_buffer("_positions_h", grid_h.flatten())
        self.register_buffer("_positions_w", grid_w.flatten())

        self._freqs_t: Tensor
        self._freqs_h: Tensor
        self._freqs_w: Tensor
        self._positions_t: Tensor
        self._positions_h: Tensor
        self._positions_w: Tensor

    @override
    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply 3D RoPE to query and key tensors.

        Args:
            q: Query tensor of shape [batch, heads, seq_len, head_dim].
            k: Key tensor of shape [batch, heads, seq_len, head_dim].

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        # Get frequencies for each position
        freqs_t = self._freqs_t[self._positions_t]
        freqs_h = self._freqs_h[self._positions_h]
        freqs_w = self._freqs_w[self._positions_w]

        # Split along head dimension
        q_t, q_h, q_w = q.split([self._dim_t, self._dim_h, self._dim_w], dim=-1)
        k_t, k_h, k_w = k.split([self._dim_t, self._dim_h, self._dim_w], dim=-1)

        # Apply rotary embeddings
        q_rot = torch.cat(
            [
                _apply_rotary_emb(q_t, freqs_t),
                _apply_rotary_emb(q_h, freqs_h),
                _apply_rotary_emb(q_w, freqs_w),
            ],
            dim=-1,
        )

        k_rot = torch.cat(
            [
                _apply_rotary_emb(k_t, freqs_t),
                _apply_rotary_emb(k_h, freqs_h),
                _apply_rotary_emb(k_w, freqs_w),
            ],
            dim=-1,
        )

        return q_rot, k_rot

    __call__: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
