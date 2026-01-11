import pytest
import torch

from exp.trainer.vjepa import MaskConfig, VideoMultiBlockMaskCollator


class TestMaskConfig:
    @pytest.mark.parametrize(
        "num_blocks, spatial_scale",
        [
            (1, 0.15),
            (8, 0.7),
            (10, 1.0),
            (5, 0.5),
        ],
    )
    def test_valid_config(self, num_blocks, spatial_scale):
        config = MaskConfig(num_blocks=num_blocks, spatial_scale=spatial_scale)
        assert config.num_blocks == num_blocks
        assert config.spatial_scale == spatial_scale

    @pytest.mark.parametrize(
        "num_blocks, spatial_scale, match",
        [
            (0, 0.5, "num_blocks must be at least 1"),
            (-1, 0.5, "num_blocks must be at least 1"),
            (5, 0.0, "spatial_scale must be in \\(0, 1\\]"),
            (5, -0.1, "spatial_scale must be in \\(0, 1\\]"),
            (5, 1.1, "spatial_scale must be in \\(0, 1\\]"),
        ],
    )
    def test_invalid_config(self, num_blocks, spatial_scale, match):
        with pytest.raises(ValueError, match=match):
            MaskConfig(num_blocks=num_blocks, spatial_scale=spatial_scale)


class TestVideoMultiBlockMaskCollator:
    @pytest.mark.parametrize(
        "num_tubelets, aspect_ratio, min_keep",
        [
            ((2, 4, 4), (0.75, 1.5), 4),
            ((4, 8, 8), (0.5, 2.0), 10),
            ((2, 3, 3), (1.0, 1.0), 2),
        ],
    )
    def test_init_valid(self, num_tubelets, aspect_ratio, min_keep):
        collator = VideoMultiBlockMaskCollator(
            num_tubelets=num_tubelets,
            aspect_ratio=aspect_ratio,
            min_keep=min_keep,
        )
        assert (
            collator.n_tubelets == num_tubelets[0] * num_tubelets[1] * num_tubelets[2]
        )

    @pytest.mark.parametrize(
        "num_tubelets, aspect_ratio, min_keep, match",
        [
            (
                (2, 4, 4),
                (1.5, 0.75),
                4,
                "aspect_ratio\\[0\\] must be <= aspect_ratio\\[1\\]",
            ),
            ((2, 4, 4), (0.0, 1.5), 4, "aspect_ratio\\[0\\] must be positive"),
            ((2, 4, 4), (-0.5, 1.5), 4, "aspect_ratio\\[0\\] must be positive"),
            ((2, 4, 4), (0.75, 1.5), 0, "min_keep must be at least 1"),
            ((2, 4, 4), (0.75, 1.5), -5, "min_keep must be at least 1"),
            ((2, 2, 2), (0.75, 1.5), 4, "must exceed min_keep"),
            ((2, 2, 2), (0.75, 1.5), 10, "must exceed min_keep"),
        ],
    )
    def test_init_invalid(self, num_tubelets, aspect_ratio, min_keep, match):
        with pytest.raises(ValueError, match=match):
            VideoMultiBlockMaskCollator(
                num_tubelets=num_tubelets,
                aspect_ratio=aspect_ratio,
                min_keep=min_keep,
            )

    @pytest.mark.parametrize(
        "num_tubelets, expected",
        [
            ((2, 4, 4), 32),
            ((3, 5, 6), 90),
            ((4, 10, 10), 400),
        ],
    )
    def test_n_tubelets_property(self, num_tubelets, expected):
        collator = VideoMultiBlockMaskCollator(num_tubelets=num_tubelets)
        assert collator.n_tubelets == expected

    def test_sample_masks_and_target(self):
        collator = VideoMultiBlockMaskCollator(num_tubelets=(2, 8, 8))
        generator = torch.Generator()
        generator.manual_seed(42)

        encoder_mask, predictor_target = collator.sample_masks_and_target(generator)

        assert encoder_mask.shape == (2 * 8 * 8,)
        assert predictor_target.shape == (2 * 8 * 8,)
        assert encoder_mask.dtype == torch.bool
        assert predictor_target.dtype == torch.bool
        assert encoder_mask.any()
        assert predictor_target.any()

    def test_sample_masks_and_target_respects_min_keep(self):
        collator = VideoMultiBlockMaskCollator(num_tubelets=(2, 4, 4), min_keep=10)
        generator = torch.Generator()
        generator.manual_seed(42)

        encoder_mask, _ = collator.sample_masks_and_target(generator)

        # Check that at least min_keep spatial positions are unmasked
        spatial_size = 4 * 4
        unmasked_count = (~encoder_mask[:spatial_size]).sum().item()
        assert unmasked_count >= 10

    def test_call_single_video(self):
        collator = VideoMultiBlockMaskCollator(num_tubelets=(2, 4, 4))
        video = torch.randn(3, 4, 32, 32)  # [C, T, H, W]

        videos, encoder_masks, predictor_targets = collator([video])

        assert videos.shape == (1, 3, 4, 32, 32)
        assert encoder_masks.shape == (1, 2 * 4 * 4)
        assert predictor_targets.shape == (1, 2 * 4 * 4)
        assert encoder_masks.dtype == torch.bool
        assert predictor_targets.dtype == torch.bool

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_call_multiple_videos(self, batch_size):
        collator = VideoMultiBlockMaskCollator(num_tubelets=(2, 4, 4))
        videos_list = [torch.randn(3, 4, 32, 32) for _ in range(batch_size)]

        videos, encoder_masks, predictor_targets = collator(videos_list)

        assert videos.shape == (batch_size, 3, 4, 32, 32)
        assert encoder_masks.shape == (batch_size, 2 * 4 * 4)
        assert predictor_targets.shape == (batch_size, 2 * 4 * 4)

    def test_call_with_tuples(self):
        collator = VideoMultiBlockMaskCollator(num_tubelets=(2, 4, 4))
        video = torch.randn(3, 4, 32, 32)
        videos_list = [(video, video), (video, video)]

        videos, encoder_masks, predictor_targets = collator(videos_list)

        assert videos.shape == (2, 3, 4, 32, 32)
        assert encoder_masks.shape == (2, 2 * 4 * 4)
        assert predictor_targets.shape == (2, 2 * 4 * 4)

    def test_different_masks_per_sample(self):
        collator = VideoMultiBlockMaskCollator(num_tubelets=(2, 4, 4))
        videos_list = [
            torch.randn(3, 4, 32, 32),
            torch.randn(3, 4, 32, 32),
        ]

        _, encoder_masks, predictor_targets = collator(videos_list)

        # Masks should be different for different samples
        assert not torch.equal(encoder_masks[0], encoder_masks[1])
        assert not torch.equal(predictor_targets[0], predictor_targets[1])
