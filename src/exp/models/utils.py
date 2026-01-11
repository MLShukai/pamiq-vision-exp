import math

import torch.nn as nn


def init_weights(module: nn.Module, init_std: float = 0.02) -> None:
    """Initialize module weights using truncated normal distribution.

    Args:
        module: PyTorch module to initialize.
        init_std: Standard deviation for truncated normal initialization.
    """
    match module:
        case nn.Linear():
            nn.init.trunc_normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        case nn.Conv3d() | nn.Conv2d() | nn.ConvTranspose3d() | nn.ConvTranspose2d():
            nn.init.trunc_normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        case nn.LayerNorm():
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def rescale_weight_for_depth(module: nn.Module, depth: int) -> None:
    """Rescale module weights by depth factor for training stability.

    Args:
        module: Module whose weights to rescale.
        depth: Current depth (starting from 1).

    Raises:
        ValueError: If depth is zero.
    """
    if depth == 0:
        raise ValueError("Depth must be non-zero.")
    factor = math.sqrt(2.0 * depth)
    match module:
        case nn.Linear():
            module.weight.data.div_(factor)
