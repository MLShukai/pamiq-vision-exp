import math

import pytest
import torch
import torch.nn as nn

from exp.models.utils import init_weights, rescale_weight_for_depth


class TestInitWeights:
    @pytest.mark.parametrize(
        "module_class, has_bias",
        [
            (nn.Linear, True),
            (nn.Linear, False),
        ],
    )
    def test_init_weights_linear(self, module_class, has_bias):
        module = module_class(10, 20, bias=has_bias)
        init_weights(module, init_std=0.02)

        # Weight should be initialized (not all zeros)
        assert not torch.allclose(module.weight, torch.zeros_like(module.weight))

        # Bias should be zeros if it exists
        if has_bias:
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    @pytest.mark.parametrize(
        "module_class, has_bias",
        [
            (nn.Conv3d, True),
            (nn.Conv3d, False),
            (nn.Conv2d, True),
            (nn.Conv2d, False),
            (nn.ConvTranspose3d, True),
            (nn.ConvTranspose3d, False),
            (nn.ConvTranspose2d, True),
            (nn.ConvTranspose2d, False),
        ],
    )
    def test_init_weights_conv(self, module_class, has_bias):
        if "3d" in module_class.__name__.lower():
            module = module_class(3, 16, kernel_size=3, bias=has_bias)
        else:
            module = module_class(3, 16, kernel_size=3, bias=has_bias)

        init_weights(module, init_std=0.02)

        # Weight should be initialized (not all zeros)
        assert not torch.allclose(module.weight, torch.zeros_like(module.weight))

        # Bias should be zeros if it exists
        if has_bias:
            assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    def test_init_weights_layernorm(self):
        module = nn.LayerNorm(10)
        init_weights(module, init_std=0.02)

        # Weight should be ones
        assert torch.allclose(module.weight, torch.ones_like(module.weight))

        # Bias should be zeros
        assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    def test_init_weights_unsupported_module(self):
        # Test that unsupported modules don't raise errors
        module = nn.ReLU()
        init_weights(module, init_std=0.02)  # Should not raise


class TestRescaleWeightForDepth:
    @pytest.mark.parametrize(
        "depth",
        [1, 2, 5, 10],
    )
    def test_rescale_weight_for_depth_linear(self, depth):
        module = nn.Linear(10, 20)
        original_weight = module.weight.data.clone()

        rescale_weight_for_depth(module, depth)

        expected_factor = math.sqrt(2.0 * depth)
        expected_weight = original_weight / expected_factor

        assert torch.allclose(module.weight.data, expected_weight)

    def test_rescale_weight_for_depth_zero_raises(self):
        module = nn.Linear(10, 20)

        with pytest.raises(ValueError, match="Depth must be non-zero"):
            rescale_weight_for_depth(module, 0)

    def test_rescale_weight_for_depth_unsupported_module(self):
        # Test that unsupported modules don't raise errors
        module = nn.ReLU()
        rescale_weight_for_depth(module, 1)  # Should not raise
