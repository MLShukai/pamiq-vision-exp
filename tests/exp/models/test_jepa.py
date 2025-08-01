import pytest
import torch

from exp.models import ModelName
from exp.models.components.image_patchifier import ImagePatchifier
from exp.models.components.positional_embeddings import get_2d_positional_embeddings
from exp.models.jepa import Encoder, LightWeightDecoder, Predictor, create_image_jepa
from exp.utils import size_2d_to_int_tuple


class TestEncoder:
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("img_size", [64, 96])
    @pytest.mark.parametrize("patch_size", [8])
    @pytest.mark.parametrize("hidden_dim", [64])
    @pytest.mark.parametrize("embed_dim", [32])
    def test_forward_without_mask(
        self, batch_size, img_size, patch_size, hidden_dim, embed_dim
    ):
        """Test Encoder's forward pass without mask."""
        n_patches = (img_size // patch_size) ** 2
        patchifier = ImagePatchifier(patch_size, 3, hidden_dim)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, (img_size // patch_size, img_size // patch_size)
        ).reshape(n_patches, hidden_dim)

        encoder = Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            depth=2,
            num_heads=2,
        )

        images = torch.randn(batch_size, 3, img_size, img_size)
        encoded = encoder(images)

        assert encoded.shape == (batch_size, n_patches, embed_dim)

    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("img_size", [64])
    @pytest.mark.parametrize("patch_size", [8])
    @pytest.mark.parametrize("mask_ratio", [0.25])
    def test_forward_with_mask(self, batch_size, img_size, patch_size, mask_ratio):
        """Test Encoder's forward pass with mask."""
        n_patches = (img_size // patch_size) ** 2
        patchifier = ImagePatchifier(patch_size, 3, 64)
        positional_encodings = get_2d_positional_embeddings(
            64, (img_size // patch_size, img_size // patch_size)
        ).reshape(n_patches, 64)

        encoder = Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=32,
            depth=2,
            num_heads=2,
        )

        images = torch.randn(batch_size, 3, img_size, img_size)

        # Create a random mask with the specified ratio
        num_mask = int(n_patches * mask_ratio)
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        for i in range(batch_size):
            mask_indices = torch.randperm(n_patches)[:num_mask]
            masks[i, mask_indices] = True

        encoded = encoder(images, masks)

        assert encoded.shape == (batch_size, n_patches, encoder.out_proj.out_features)

    def test_invalid_positional_encoding_shape(self):
        """Test error when positional encoding shape doesn't match expected
        shape."""
        patchifier = ImagePatchifier(8, 3, 64)

        with pytest.raises(
            ValueError,
            match="positional_encodings channel dimension must be hidden_dim.",
        ):
            Encoder(
                patchifier=patchifier,
                positional_encodings=torch.zeros(64, 32),  # Wrong channel size
                hidden_dim=64,
                embed_dim=32,
                depth=1,
                num_heads=2,
            )

        with pytest.raises(ValueError, match="positional_encodings must be 2d tensor!"):
            Encoder(
                patchifier=patchifier,
                positional_encodings=torch.zeros(
                    64,
                ),  # Wrong dims size
                hidden_dim=64,
                embed_dim=32,
                depth=1,
                num_heads=2,
            )

    def test_invalid_mask_shape(self):
        """Test error when mask shape doesn't match encoded image shape."""
        n_patches = 64
        patchifier = ImagePatchifier(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        images = torch.randn(2, 3, 64, 64)

        # Create mask with incorrect shape
        masks = torch.zeros(2, n_patches - 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            encoder(images, masks)

    def test_non_bool_mask(self):
        """Test error when mask tensor is not boolean."""
        n_patches = 64
        patchifier = ImagePatchifier(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        images = torch.randn(1, 3, 64, 64)

        # Create mask with incorrect dtype (float instead of bool)
        masks = torch.zeros(1, n_patches, dtype=torch.float32)

        with pytest.raises(ValueError, match="Mask tensor dtype must be bool"):
            encoder(images, masks)

    def test_clone(self):
        n_patches = 64
        patchifier = ImagePatchifier(8, 3, 64)
        positional_encodings = get_2d_positional_embeddings(64, (8, 8)).reshape(
            n_patches, 64
        )

        encoder = Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=64,
            embed_dim=64,
            depth=1,
            num_heads=2,
        )
        copied = encoder.clone()
        assert encoder is not copied
        for p, p_copied in zip(encoder.parameters(), copied.parameters(), strict=True):
            assert torch.equal(p, p_copied)


class TestPredictor:
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("n_patches", [64])
    @pytest.mark.parametrize("embed_dim", [32])
    @pytest.mark.parametrize("hidden_dim", [32])
    def test_forward(self, batch_size, n_patches, embed_dim, hidden_dim):
        """Test Predictor's forward pass."""
        positional_encodings = get_2d_positional_embeddings(hidden_dim, (8, 8)).reshape(
            n_patches, hidden_dim
        )

        predictor = Predictor(
            positional_encodings=positional_encodings,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            depth=2,
            num_heads=2,
        )

        # Create latents as if they came from encoder
        latents = torch.randn(batch_size, n_patches, embed_dim)

        # Create target mask (e.g., 25% of patches are targets)
        targets = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        for i in range(batch_size):
            target_indices = torch.randperm(n_patches)[: n_patches // 4]
            targets[i, target_indices] = True

        predictions = predictor(latents, targets)

        # Check output shape
        assert predictions.shape == (
            batch_size,
            n_patches,
            embed_dim,
        )

    def test_invalid_positional_encoding_shape(self):
        """Test error when positional encoding shape doesn't match expected
        shape."""

        with pytest.raises(
            ValueError,
            match="positional_encodings channel dimension must be hidden_dim.",
        ):
            Predictor(
                positional_encodings=torch.zeros(
                    32, 64
                ),  # Wrong shape for hidden_dim=32
                embed_dim=32,
                hidden_dim=32,
                depth=1,
                num_heads=2,
            )

        with pytest.raises(ValueError, match="positional_encodings must be 2d tensor!"):
            Predictor(
                positional_encodings=torch.zeros(
                    32,
                ),  # Wrong dim size
                embed_dim=32,
                hidden_dim=32,
                depth=1,
                num_heads=2,
            )

    def test_invalid_target_shape(self):
        """Test error when target shape doesn't match latent shape."""
        n_patches = 64
        positional_encodings = get_2d_positional_embeddings(32, (8, 8)).reshape(
            n_patches, 32
        )

        predictor = Predictor(
            positional_encodings=positional_encodings,
            embed_dim=32,
            hidden_dim=32,
            depth=1,
            num_heads=2,
        )
        latents = torch.randn(1, 64, 32)
        targets = torch.zeros(1, 32, dtype=torch.bool)  # Incorrect shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            predictor(latents, targets)

    def test_non_bool_target(self):
        """Test error when target tensor is not boolean."""
        n_patches = 64
        positional_encodings = get_2d_positional_embeddings(32, (8, 8)).reshape(
            n_patches, 32
        )

        predictor = Predictor(
            positional_encodings=positional_encodings,
            embed_dim=32,
            hidden_dim=32,
            depth=1,
            num_heads=2,
        )
        latents = torch.randn(1, 64, 32)

        # Create targets with incorrect dtype (float instead of bool)
        targets = torch.zeros(1, 64, dtype=torch.float32)

        with pytest.raises(ValueError, match="Target tensor dtype must be bool"):
            predictor(latents, targets)


class TestLightWeightDecoder:
    @pytest.mark.parametrize("n_patches", [(14, 14), 14])
    @pytest.mark.parametrize("upsample", [1, (2, 2)])
    def test_initialization(self, n_patches, upsample):
        """Test initialization with various parameters."""

        decoder = LightWeightDecoder(
            n_patches=n_patches,
            patch_size=16,
            embed_dim=768,
            out_channels=3,
            upsample=upsample,
        )

        #
        n_patches_tuple = size_2d_to_int_tuple(n_patches)
        upsample_tuple = size_2d_to_int_tuple(upsample)

        assert decoder.n_patches == n_patches_tuple
        assert decoder.upsample == upsample_tuple

    def test_invalid_upsample(self):
        """Test error when upsample factor is less than 1."""

        with pytest.raises(ValueError, match="upsample must be larger than 0"):
            LightWeightDecoder(n_patches=(14, 14), upsample=0)

        with pytest.raises(ValueError, match="upsample must be larger than 0"):
            LightWeightDecoder(n_patches=(14, 14), upsample=(0, 1))

    def test_upsampled_n_patches_property(self):
        """Test upsampled_n_patches property returns correct values."""

        decoder = LightWeightDecoder(n_patches=(14, 14), upsample=2)
        assert decoder.upsampled_n_patches == (28, 28)

        decoder = LightWeightDecoder(n_patches=14, upsample=(2, 3))
        assert decoder.upsampled_n_patches == (28, 42)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_patches", [(8, 8)])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("embed_dim", [64])
    @pytest.mark.parametrize("upsample", [1, 2])
    def test_forward_shape(
        self, batch_size, n_patches, patch_size, embed_dim, upsample
    ):
        """Test the shape of forward pass output."""
        # 期待される出力シェイプを計算
        n_patches_tuple = size_2d_to_int_tuple(n_patches)
        patch_size_tuple = size_2d_to_int_tuple(patch_size)
        upsample_tuple = size_2d_to_int_tuple(upsample)

        n_patches_after_upsample = (
            n_patches_tuple[0] * upsample_tuple[0],
            n_patches_tuple[1] * upsample_tuple[1],
        )
        expected_height = n_patches_after_upsample[0] * patch_size_tuple[0]
        expected_width = n_patches_after_upsample[1] * patch_size_tuple[1]

        decoder = LightWeightDecoder(
            n_patches=n_patches,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=3,
            upsample=upsample,
        )

        # 入力テンソルを作成
        latents = torch.randn(
            batch_size, n_patches_tuple[0] * n_patches_tuple[1], embed_dim
        )

        # 順伝播
        output = decoder(latents)

        # 出力シェイプを確認
        assert output.shape == (batch_size, 3, expected_height, expected_width)


class TestJEPAIntegration:
    def test_encoder_predictor_integration(self):
        """Test that encoder and predictor work together in a typical
        workflow."""
        # Create encoder and predictor with smaller dimensions
        img_size = 64
        patch_size = 8
        embed_dim = 32
        hidden_dim = 64

        # Calculate grid dimensions
        img_size_tuple = size_2d_to_int_tuple(img_size)
        patch_size_tuple = size_2d_to_int_tuple(patch_size)
        n_patches_h = img_size_tuple[0] // patch_size_tuple[0]
        n_patches_w = img_size_tuple[1] // patch_size_tuple[1]
        n_patches = n_patches_h * n_patches_w

        # Initialize models with reduced complexity
        patchifier = ImagePatchifier(patch_size, 3, hidden_dim)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, (n_patches_h, n_patches_w)
        ).reshape(n_patches, hidden_dim)

        # Initialize models with reduced complexity
        encoder = Encoder(
            patchifier=patchifier,
            positional_encodings=positional_encodings,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            depth=2,
            num_heads=2,
        )

        predictor = Predictor(
            positional_encodings=positional_encodings[
                :, :32
            ],  # Use first 32 dims for predictor
            embed_dim=embed_dim,
            hidden_dim=32,
            depth=1,
            num_heads=2,
        )

        decoder = LightWeightDecoder(
            n_patches=(n_patches_h, n_patches_w),
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Create a smaller batch of images
        batch_size = 1
        images = torch.randn(batch_size, 3, img_size, img_size)

        # Create context and target masks (non-overlapping)
        context_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        target_mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        target_mask[:, n_patches // 2 :] = True

        # Encode with context mask
        encoded = encoder(images, context_mask)

        # Predict with target mask
        predictions = predictor(encoded, target_mask)

        # Decode
        decoded = decoder(encoded)

        # Check shapes
        assert encoded.shape == (batch_size, n_patches, embed_dim)
        assert predictions.shape == (batch_size, n_patches, embed_dim)
        assert decoded.shape == (batch_size, 3, img_size, img_size)


class TestCreateImageJEPA:
    @pytest.mark.parametrize(
        "image_size,patch_size",
        [
            (64, 8),
            (224, 16),
            ((96, 128), (12, 16)),
            (512, 32),
        ],
    )
    def test_create_image_jepa_objects(
        self,
        image_size,
        patch_size,
    ):
        """Test that create_image_jepa creates objects of correct types."""
        models = create_image_jepa(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            hidden_dim=256,
            embed_dim=128,
            depth=2,
            num_heads=2,
        )

        # Check object types
        assert isinstance(models[ModelName.JEPA_CONTEXT_ENCODER], Encoder)
        assert isinstance(models[ModelName.JEPA_TARGET_ENCODER], Encoder)
        assert isinstance(models[ModelName.JEPA_PREDICTOR], Predictor)

        # Check that target encoder is a separate instance (cloned)
        assert (
            models[ModelName.JEPA_CONTEXT_ENCODER]
            is not models[ModelName.JEPA_TARGET_ENCODER]
        )

        # Check that parameters are initially identical (cloned properly)
        for ctx_param, tgt_param in zip(
            models[ModelName.JEPA_CONTEXT_ENCODER].parameters(),
            models[ModelName.JEPA_TARGET_ENCODER].parameters(),
        ):
            assert torch.equal(ctx_param, tgt_param)
