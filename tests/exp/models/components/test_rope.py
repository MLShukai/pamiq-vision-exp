import pytest
import torch

from exp.models.components.rope import RoPE3D


class TestRoPE3D:
    @pytest.mark.parametrize(
        "batch_size, num_heads, n_temporal, n_height, n_width, head_dim",
        [
            (2, 8, 8, 14, 14, 96),
            (1, 12, 4, 7, 7, 64),
            (4, 16, 16, 10, 10, 48),
        ],
    )
    def test_forward(
        self, batch_size, num_heads, n_temporal, n_height, n_width, head_dim
    ):
        rope = RoPE3D(
            dim=head_dim,
            n_temporal=n_temporal,
            n_height=n_height,
            n_width=n_width,
        )

        seq_len = n_temporal * n_height * n_width
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_changes_values(self):
        rope = RoPE3D(dim=96, n_temporal=8, n_height=14, n_width=14)

        seq_len = 8 * 14 * 14
        q = torch.randn(2, 8, seq_len, 96)
        k = torch.randn(2, 8, seq_len, 96)

        q_rot, k_rot = rope(q, k)

        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)
