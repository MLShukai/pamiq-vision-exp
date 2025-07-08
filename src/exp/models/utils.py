"""Utility tools for model definitions."""

import torch.nn as nn

type size_2d = int | tuple[int, int]


def size_2d_to_int_tuple(size: size_2d) -> tuple[int, int]:
    """Convert `size_2d` type to int tuple."""
    return (size, size) if isinstance(size, int) else (size[0], size[1])


def init_weights(m: nn.Module, init_std: float) -> None:
    """Initialize the weights with truncated normal distribution and zeros for
    biases.

    Args:
        m: Module to initialize.
        init_std: Standard deviation for the truncated normal initialization.
    """
    match m:
        case nn.Linear() | nn.Conv2d() | nn.ConvTranspose2d():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case nn.LayerNorm():
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        case _:
            pass
