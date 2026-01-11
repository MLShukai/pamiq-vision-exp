import pytest
import torch

from exp.models.components.rope import RoPE3D
from exp.models.components.transformer import (
    MLP,
    Attention,
    Transformer,
    TransformerLayer,
)


class TestMLP:
    @pytest.mark.parametrize(
        "batch_size, seq_len, in_features, hidden_features, out_features",
        [
            (2, 10, 64, 256, 64),
            (1, 5, 32, 128, 32),
        ],
    )
    def test_forward(
        self, batch_size, seq_len, in_features, hidden_features, out_features
    ):
        mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
        )

        x = torch.randn(batch_size, seq_len, in_features)
        output = mlp(x)

        assert output.shape == (batch_size, seq_len, out_features)


class TestAttention:
    @pytest.mark.parametrize(
        "batch_size, seq_len, dim, num_heads, n_temporal, n_height, n_width",
        [
            (2, 8, 64, 4, 2, 2, 2),
            (1, 12, 48, 4, 2, 3, 2),
        ],
    )
    def test_forward(
        self, batch_size, seq_len, dim, num_heads, n_temporal, n_height, n_width
    ):
        attn = Attention(dim=dim, num_heads=num_heads)

        rope = RoPE3D(
            dim=dim // num_heads,
            n_temporal=n_temporal,
            n_height=n_height,
            n_width=n_width,
        )

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x, rope)

        assert output.shape == (batch_size, seq_len, dim)


class TestTransformerLayer:
    @pytest.mark.parametrize(
        "batch_size, seq_len, embedding_dim, num_heads, n_temporal, n_height, n_width",
        [
            (2, 8, 64, 4, 2, 2, 2),
            (1, 12, 48, 4, 2, 3, 2),
        ],
    )
    def test_forward(
        self,
        batch_size,
        seq_len,
        embedding_dim,
        num_heads,
        n_temporal,
        n_height,
        n_width,
    ):
        layer = TransformerLayer(embedding_dim=embedding_dim, num_heads=num_heads)

        rope = RoPE3D(
            dim=embedding_dim // num_heads,
            n_temporal=n_temporal,
            n_height=n_height,
            n_width=n_width,
        )

        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = layer(x, rope)

        assert output.shape == (batch_size, seq_len, embedding_dim)


class TestTransformer:
    @pytest.mark.parametrize(
        "batch_size, embedding_dim, depth, num_heads, n_temporal, n_height, n_width",
        [
            (2, 64, 2, 4, 2, 2, 2),
            (1, 48, 2, 4, 2, 3, 2),
        ],
    )
    def test_forward(
        self,
        batch_size,
        embedding_dim,
        depth,
        num_heads,
        n_temporal,
        n_height,
        n_width,
    ):
        transformer = Transformer(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            n_temporal=n_temporal,
            n_height=n_height,
            n_width=n_width,
        )

        seq_len = n_temporal * n_height * n_width
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = transformer(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)
