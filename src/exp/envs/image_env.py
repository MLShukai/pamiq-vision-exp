from collections.abc import Callable
from typing import Any, override

from pamiq_core import Environment
from torch import Tensor


class ImageEnvironment(Environment[Tensor, Any]):
    """Environment that generates and returns image tensors."""

    def __init__(
        self,
        image_generator: Callable[[], Tensor],
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize the ImageEnvironment.

        Args:
            image_generator: Function that returns image tensors
            transform: Optional transform to apply to generated images
        """
        super().__init__()

        self._generator = image_generator
        self._transform = transform

    @override
    def observe(self) -> Tensor:
        """Generate and process an image observation.

        Returns:
            Processed image tensor

        Raises:
            ValueError: If the generated image doesn't have 3 dimensions [C, H, W]
        """
        image = Tensor(self._generator())
        if self._transform is not None:
            image = self._transform(image)

        if image.ndim != 3:
            raise ValueError(
                f"Generated image must have 3 dimensions [C, H, W], got shape {image.shape}"
            )

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
