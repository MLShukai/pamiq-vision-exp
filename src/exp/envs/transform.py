from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    ToDtype,
    ToImage,
    ToPureTensor,
)

from exp.utils import Size2D, size_2d_to_int_tuple


def standardize(tensor: Tensor) -> Tensor:
    """Standardize tensor to zero mean and unit variance."""
    return (tensor - tensor.mean()) / (tensor.std() + 1e-6)


def create_transform(
    size: Size2D,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Callable[[Any], Tensor]:
    """Create image transform pipeline.

    Args:
        size: Target size (height, width) for resizing.
        device: Target device for tensors. Defaults to current device.
        dtype: Target dtype for tensors. Defaults to current dtype.

    Returns:
        Transform function that converts images to standardized tensors.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()

    def to_device(tensor: Tensor) -> Tensor:
        return tensor.to(device)

    return Compose(
        [
            ToImage(),
            to_device,
            ToDtype(dtype, scale=True),
            Resize(size_2d_to_int_tuple(size)),
            standardize,
            ToPureTensor(),
        ]
    )
