from functools import partial
from pathlib import Path

import pytest
import torch
from pamiq_core.data.impls import RandomReplacementBuffer
from pamiq_core.testing import connect_components
from pamiq_core.torch import TorchTrainingModel
from pytest_mock import MockerFixture
from torch.optim import AdamW

from exp.buffers import BufferName
from exp.models.jepa import create_image_jepa
from exp.trainers.jepa import (
    JEPATrainer,
    MultiBlockMaskCollator2d,
)
from tests.helpers import parametrize_device


class TestJEPATrainer:
    IMAGE_SIZE = 64
    PATCH_SIZE = 8
    CHANNELS = 3
    EMBED_DIM = 16
    HIDDEN_DIM = 128
    N_PATCHES = IMAGE_SIZE // PATCH_SIZE

    @pytest.fixture
    def models(self):
        return create_image_jepa(
            image_size=self.IMAGE_SIZE,
            patch_size=self.PATCH_SIZE,
            in_channels=self.CHANNELS,
            hidden_dim=self.HIDDEN_DIM,
            embed_dim=self.EMBED_DIM,
            depth=1,
            num_heads=2,
        )

    @pytest.fixture
    def data_buffers(self):
        return {BufferName.IMAGE: RandomReplacementBuffer(max_size=16)}

    @pytest.fixture
    def partial_optimizer(self):
        partial_optimizer = partial(AdamW, lr=1e-4, weight_decay=0.04)
        return partial_optimizer

    @pytest.fixture
    def collate_fn(self) -> MultiBlockMaskCollator2d:
        return MultiBlockMaskCollator2d(
            num_patches=self.N_PATCHES,
        )

    @pytest.fixture
    def trainer(self, partial_optimizer, collate_fn, mocker: MockerFixture):
        mock_run = mocker.MagicMock()
        mocker.patch("exp.trainers.base.get_global_run", return_value=mock_run)
        return JEPATrainer(
            partial_optimizer,
            collate_fn=collate_fn,
            batch_size=2,
            min_buffer_size=4,
            min_new_data_count=2,
        )

    @parametrize_device
    def test_run(self, device, data_buffers, models, trainer: JEPATrainer):
        """Test JEPA Trainer workflow."""
        models = {
            name: TorchTrainingModel(m, has_inference_model=False, device=device)
            for name, m in models.items()
        }

        components = connect_components(
            trainers=trainer, buffers=data_buffers, models=models
        )
        collector = components.data_collectors[BufferName.IMAGE]
        for _ in range(10):
            collector.collect(
                torch.randn(self.CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE)
            )

        assert trainer.global_steps == 0
        assert trainer.run() is True
        assert trainer.global_steps > 0
        global_steps = trainer.global_steps
        assert trainer.run() is False
        assert trainer.global_steps == global_steps


class TestMultiBlockMaskCollator2d:
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("min_keep", [10])
    @pytest.mark.parametrize("mask_scale", [(0.1, 0.25)])
    def test_sample_mask_rectangle(self, image_size, patch_size, min_keep, mask_scale):
        """Test that the sampled mask rectangle has valid dimensions and
        follows constraints."""
        collator = MultiBlockMaskCollator2d(
            num_patches=image_size // patch_size,
            mask_scale=mask_scale,
            min_keep=min_keep,
        )
        g = torch.Generator()
        n_patches = (image_size // patch_size) ** 2

        for _ in range(100):
            top, bottom, left, right = collator._sample_mask_rectangle(g)

            # Check coordinates are valid
            assert top < bottom
            assert top >= 0
            assert bottom <= collator.n_patches_height
            assert left < right
            assert left >= 0
            assert right <= collator.n_patches_width

            # Calculate mask dimensions
            height, width = (bottom - top), (right - left)
            mask_area = height * width

            # Test mask scale
            mask_scale_min, mask_scale_max = mask_scale
            assert mask_area <= mask_scale_max * n_patches
            assert mask_area >= mask_scale_min * n_patches

            # Test min keep
            assert mask_area >= min_keep

    @pytest.mark.parametrize("image_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("n_masks", [4])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_collator_call(
        self,
        image_size: int,
        patch_size: int,
        n_masks: int,
        batch_size: int,
    ):
        """Test the collator's __call__ method for end-to-end functionality."""
        assert image_size % patch_size == 0

        # Initialize collator
        collator = MultiBlockMaskCollator2d(
            num_patches=(image_size // patch_size, image_size // patch_size),
            n_masks=n_masks,
            min_keep=50,
        )

        # Create sample inputs
        images = [
            (torch.randn([3, image_size, image_size]),) for _ in range(batch_size)
        ]

        # Call collator
        (
            collated_images,
            collated_encoder_masks,
            collated_predictor_targets,
        ) = collator(images)

        # Check image sizes
        assert collated_images.size(0) == batch_size, "batch_size mismatch"
        assert collated_images.size(1) == 3, "channel mismatch"
        assert collated_images.size(2) == image_size, "collated_images height mismatch"
        assert collated_images.size(3) == image_size, "collated_images width mismatch"

        # Calculate number of patches
        n_patches = collator.n_patches

        # Check encoder masks
        assert collated_encoder_masks.dim() == 2
        assert (
            collated_encoder_masks.size(0) == batch_size
        ), "batch_size mismatch (collated_encoder_masks)"
        assert (
            collated_encoder_masks.size(1) == n_patches
        ), "patch count mismatch (collated_encoder_masks)"
        assert (
            collated_encoder_masks.dtype == torch.bool
        ), "dtype mismatch (collated_encoder_masks)"

        # Check predictor targets
        assert collated_predictor_targets.dim() == 2
        assert (
            collated_predictor_targets.size(0) == batch_size
        ), "batch_size mismatch (collated_predictor_targets)"
        assert (
            collated_predictor_targets.size(1) == n_patches
        ), "patch count mismatch (collated_predictor_targets)"
        assert (
            collated_predictor_targets.dtype == torch.bool
        ), "dtype mismatch (collated_predictor_targets)"

        # Check that at least min_keep patches are unmasked for encoder
        assert (
            torch.sum(~collated_encoder_masks, dim=1).min() >= collator.min_keep
        ), "min_keep not satisfied for encoder"

        # Check that at least one patch is masked for predictor target
        assert (
            torch.sum(collated_predictor_targets, dim=1).min() > 0
        ), "no prediction target for predictor"

        # Check that encoder masks and predictor targets are not identical
        assert not torch.all(
            collated_encoder_masks == collated_predictor_targets
        ), "encoder masks and predictor targets must be different"

    def test_sample_masks_and_target(self):
        """Test the sample_masks_and_target method for correct output shapes
        and properties."""
        image_size, patch_size = 224, 16
        collator = MultiBlockMaskCollator2d(
            num_patches=(image_size // patch_size, image_size // patch_size),
            n_masks=4,
            min_keep=50,
        )

        g = torch.Generator()
        encoder_mask, predictor_target = collator.sample_masks_and_target(g)

        n_patches = collator.n_patches

        # Check shapes
        assert encoder_mask.shape == (n_patches,)
        assert predictor_target.shape == (n_patches,)

        # Check dtypes
        assert encoder_mask.dtype == torch.bool
        assert predictor_target.dtype == torch.bool

        # Check that at least min_keep patches are unmasked for encoder
        assert torch.sum(~encoder_mask) >= collator.min_keep

        # Check that at least one patch is masked for predictor target
        assert torch.sum(predictor_target) > 0

        # Check that encoder mask and predictor target are not identical
        assert not torch.all(encoder_mask == predictor_target)

    def test_n_patches_property(self):
        """Test that the n_patches property returns the correct value."""
        image_size, patch_size = 224, 16
        collator = MultiBlockMaskCollator2d(
            num_patches=(image_size // patch_size, image_size // patch_size),
        )

        expected_patches = (image_size // patch_size) ** 2
        assert collator.n_patches == expected_patches

    def test_step_method(self):
        """Test that the step method increments the counter properly."""
        collator = MultiBlockMaskCollator2d(
            num_patches=224 // 16,
        )

        # Get initial value
        initial_value = collator.step()

        # Check that step increments
        assert collator.step() == initial_value + 1
        assert collator.step() == initial_value + 2

    @pytest.mark.parametrize(
        "mask_scale,expected_error",
        [
            ((0.3, 0.2), "mask_scale\\[0\\] must be less than mask_scale\\[1\\]"),
            ((-0.1, 0.2), "mask_scale\\[0\\] must be greater than 0"),
            ((0.1, 1.1), "mask_scale\\[1\\] must be less than 1"),
        ],
    )
    def test_invalid_mask_scale(self, mask_scale, expected_error):
        """Test error when mask_scale is invalid."""
        with pytest.raises(ValueError, match=expected_error):
            MultiBlockMaskCollator2d(
                num_patches=224 // 16,
                mask_scale=mask_scale,
            )

    def test_min_keep_too_large(self):
        """Test error when min_keep is larger than total patches."""
        with pytest.raises(
            ValueError, match="min_keep .* must be less than or equal to total patches"
        ):
            MultiBlockMaskCollator2d(
                num_patches=224 // 16,
                min_keep=1000,  # Much larger than available patches
            )
