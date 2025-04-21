import pytest
import torch

from pamiq_vision_exp.models.components.patch_decoder import PatchDecoder
from pamiq_vision_exp.models.components.patch_embedding import PatchEmbedding


class TestPatchDecoder:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("img_size", [224, 384])
    @pytest.mark.parametrize("patch_size", [16, 32])
    @pytest.mark.parametrize("embed_dim", [768])
    def test_forward(self, batch_size, img_size, patch_size, embed_dim):
        """Test forward pass of the PatchDecoder."""
        n_patches_1d = img_size // patch_size
        n_patches = (n_patches_1d, n_patches_1d)

        decoder = PatchDecoder(
            n_patches=n_patches,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=3,
        )

        # Create dummy patch embeddings
        x = torch.randn(batch_size, n_patches_1d * n_patches_1d, embed_dim)

        # Forward pass
        output = decoder(x)

        # Check output shape
        assert output.shape == (batch_size, 3, img_size, img_size)

    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("img_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("embed_dim", [768])
    def test_embedding_decoding_cycle(
        self, batch_size, img_size, patch_size, embed_dim
    ):
        """Test embedding followed by decoding approximates identity
        function."""
        # Create random input image
        input_img = torch.randn(batch_size, 3, img_size, img_size)

        # Create embedding and decoding layers
        embedder = PatchEmbedding(
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )

        n_patches_1d = img_size // patch_size
        decoder = PatchDecoder(
            n_patches=(n_patches_1d, n_patches_1d),
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=3,
        )

        # Forward pass through embedding and decoding
        embeddings = embedder(input_img)
        output_img = decoder(embeddings)

        # Check shapes match
        assert output_img.shape == input_img.shape
