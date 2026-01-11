import pytest
import torch

from exp.models.components.patchfier import VideoPatchifier
from exp.models.vjepa import Encoder, LightWeightDecoder, Predictor, create_video_jepa


class TestEncoder:
    @pytest.mark.parametrize(
        "batch_size, n_tubelets, hidden_dim, embed_dim",
        [
            (2, (2, 2, 2), 64, 32),
            (1, (2, 3, 2), 48, 24),
        ],
    )
    def test_forward_without_mask(self, batch_size, n_tubelets, hidden_dim, embed_dim):
        encoder = Encoder(
            n_tubelets=n_tubelets,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            depth=2,
            num_heads=4,
        )

        n_t, n_h, n_w = n_tubelets
        total_tubelets = n_t * n_h * n_w
        video = torch.randn(batch_size, total_tubelets, hidden_dim)

        output = encoder(video)

        assert output.shape == (batch_size, total_tubelets, embed_dim)

    def test_forward_with_mask(self):
        n_tubelets = (2, 2, 2)
        encoder = Encoder(
            n_tubelets=n_tubelets, hidden_dim=64, embed_dim=32, depth=2, num_heads=4
        )

        batch_size = 2
        n_t, n_h, n_w = n_tubelets
        total_tubelets = n_t * n_h * n_w
        video = torch.randn(batch_size, total_tubelets, 64)

        masks = torch.zeros(batch_size, total_tubelets, dtype=torch.bool)
        masks[:, :2] = True

        output = encoder(video, masks)

        assert output.shape == (batch_size, total_tubelets, 32)

    def test_forward_raises_on_wrong_mask_shape(self):
        encoder = Encoder(
            n_tubelets=(2, 2, 2), hidden_dim=64, embed_dim=32, depth=2, num_heads=4
        )

        video = torch.randn(2, 8, 64)
        wrong_masks = torch.zeros(2, 4, dtype=torch.bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            encoder(video, wrong_masks)

    def test_forward_raises_on_wrong_mask_dtype(self):
        encoder = Encoder(
            n_tubelets=(2, 2, 2), hidden_dim=64, embed_dim=32, depth=2, num_heads=4
        )

        video = torch.randn(2, 8, 64)
        wrong_masks = torch.zeros(2, 8, dtype=torch.float32)

        with pytest.raises(ValueError, match="dtype must be bool"):
            encoder(video, wrong_masks)


class TestPredictor:
    @pytest.mark.parametrize(
        "batch_size, n_tubelets, embed_dim, hidden_dim",
        [
            (2, (2, 2, 2), 32, 64),
            (1, (2, 3, 2), 24, 48),
        ],
    )
    def test_forward(self, batch_size, n_tubelets, embed_dim, hidden_dim):
        predictor = Predictor(
            n_tubelets=n_tubelets,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            depth=2,
            num_heads=4,
        )

        n_t, n_h, n_w = n_tubelets
        total_tubelets = n_t * n_h * n_w
        latents = torch.randn(batch_size, total_tubelets, embed_dim)
        targets = torch.zeros(batch_size, total_tubelets, dtype=torch.bool)
        targets[:, :2] = True

        output = predictor(latents, targets)

        assert output.shape == (batch_size, total_tubelets, embed_dim)

    def test_forward_raises_on_wrong_target_shape(self):
        predictor = Predictor(
            n_tubelets=(2, 2, 2), embed_dim=32, hidden_dim=64, depth=2, num_heads=4
        )

        latents = torch.randn(2, 8, 32)
        wrong_targets = torch.zeros(2, 4, dtype=torch.bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            predictor(latents, wrong_targets)

    def test_forward_raises_on_wrong_target_dtype(self):
        predictor = Predictor(
            n_tubelets=(2, 2, 2), embed_dim=32, hidden_dim=64, depth=2, num_heads=4
        )

        latents = torch.randn(2, 8, 32)
        wrong_targets = torch.zeros(2, 8, dtype=torch.float32)

        with pytest.raises(ValueError, match="dtype must be bool"):
            predictor(latents, wrong_targets)


class TestLightWeightDecoder:
    @pytest.mark.parametrize(
        "batch_size, n_tubelets, tubelet_size, embed_dim",
        [
            (2, (2, 2, 2), (2, 8, 8), 64),
            (1, (2, 3, 2), (2, 8, 8), 48),
        ],
    )
    def test_forward(self, batch_size, n_tubelets, tubelet_size, embed_dim):
        decoder = LightWeightDecoder(
            n_tubelets=n_tubelets, tubelet_size=tubelet_size, embed_dim=embed_dim
        )

        n_t, n_h, n_w = n_tubelets
        total_tubelets = n_t * n_h * n_w
        latents = torch.randn(batch_size, total_tubelets, embed_dim)

        output = decoder(latents)

        expected_time = n_t * tubelet_size[0]
        expected_height = n_h * tubelet_size[1]
        expected_width = n_w * tubelet_size[2]

        assert output.shape == (
            batch_size,
            3,
            expected_time,
            expected_height,
            expected_width,
        )

    def test_upsampled_n_tubelets_property(self):
        decoder = LightWeightDecoder(
            n_tubelets=(2, 2, 2), tubelet_size=(2, 8, 8), embed_dim=64, upsample=2
        )

        assert decoder.upsampled_n_tubelets == (4, 4, 4)

    def test_raises_on_invalid_upsample(self):
        with pytest.raises(ValueError, match="upsample must be >= 1"):
            LightWeightDecoder(
                n_tubelets=(2, 2, 2),
                tubelet_size=(2, 8, 8),
                embed_dim=64,
                upsample=(0, 1, 1),
            )


class TestCreateVideoJepa:
    def test_create_video_jepa(self):
        video_shape = (16, 64, 64)
        tubelet_size = (2, 8, 8)

        models = create_video_jepa(
            video_shape=video_shape,
            tubelet_size=tubelet_size,
            hidden_dim=64,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )

        assert "context_encoder" in models
        assert "target_encoder" in models
        assert "predictor" in models

        assert isinstance(models["context_encoder"], Encoder)
        assert isinstance(models["target_encoder"], Encoder)
        assert isinstance(models["predictor"], Predictor)
