from collections.abc import Callable
from typing import Any, override

import torch
import torchvision.transforms.v2.functional as F
from pamiq_core import Environment
from torch import Tensor

from exp.utils import size_2d, size_2d_to_int_tuple


class ImageEnvironment(Environment[Tensor, Any]):
    """Environment that generates and returns image tensors."""

    def __init__(
        self,
        image_generator: Callable[[], Tensor],
        size: size_2d | None = None,
        standardize: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the ImageEnvironment.

        Args:
            image_generator: Function that returns image tensors
            size: Optional target size to resize images to
            standardize: Whether to standardize images (subtract mean and divide by std)
            dtype: Torch data type for the returned images
        """
        super().__init__()

        self._generator = image_generator
        if size is not None:
            self._size = list(size_2d_to_int_tuple(size))
        else:
            self._size = None
        self._standardize = standardize
        self._dtype = dtype
        self._device = device

    @override
    def observe(self) -> Tensor:
        """Generate and process an image observation.

        Returns:
            Processed image tensor

        Raises:
            ValueError: If the generated image doesn't have 3 dimensions [C, H, W]
        """
        image = self._generator()
        if image.ndim != 3:
            raise ValueError(
                f"Generated image must have 3 dimensions [C, H, W], got shape {image.shape}"
            )

        image = image.to(self._device, self._dtype)
        if self._size is not None:
            image = F.resize(image, self._size)

        if self._standardize:
            image = (image - image.mean()) / (image.std() + 1e-6)
        return image

    @override
    def affect(self, action: Any) -> None:
        """Apply action to the environment.

        This method is a no-op in this environment as the image generation
        is not affected by actions.

        Args:
            action: Ignored input action
        """
        pass
