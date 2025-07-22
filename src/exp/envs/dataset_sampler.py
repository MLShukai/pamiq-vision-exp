from collections.abc import Sized

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DatasetSampler:
    """Sequential sampler for torch datasets with random image selection.

    This class wraps torchvision datasets (e.g., CIFAR-10, CIFAR-100,
    ImageNet) and provides sequential access to a randomly selected
    subset of images.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        shuffle: bool = False,
        seed: int = 8391,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the dataset sampler.

        Args:
            dataset: A torchvision dataset instance
            seed: Random seed for reproducible sampling (default: 8391)
            max_samples: Maximum number of images to use. If None, use all images.
                        Must not exceed dataset size.

        Raises:
            ValueError: If max_samples exceeds the dataset size
        """

        self.dataset = dataset

        if not isinstance(dataset, Sized):
            raise ValueError("Dataset must implement __len__ method")

        # Get dataset size
        dataset_size = len(dataset)

        # Validate max_samples
        if max_samples is not None and max_samples > dataset_size:
            raise ValueError(
                f"max_samples ({max_samples}) exceeds dataset size ({dataset_size})"
            )

        # Determine number of samples to use
        self.num_samples = max_samples if max_samples is not None else dataset_size

        # Create and shuffle indices
        indices = np.arange(dataset_size)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

        # Select first num_samples indices
        self._selected_indices = indices[: self.num_samples]

        # Initialize current index for sequential iteration
        self._current_index = 0

    def __call__(self) -> torch.Tensor:
        """Get the next image in sequence as a torch tensor.

        Returns:
            Image tensor with shape [C, H, W] as float
        """
        # Get current dataset index
        dataset_idx = self._selected_indices[self._current_index]

        # Get image from dataset (returns (image, label) tuple)
        image, _ = self.dataset[dataset_idx]

        # Convert PIL Image to tensor if needed
        if isinstance(image, Image.Image):
            # Convert PIL to tensor [C, H, W]
            image = torch.from_numpy(np.array(image)).float()
            # If grayscale, add channel dimension
            if image.ndim == 2:
                image = image.unsqueeze(0)
            # If RGB/RGBA, move channels to first dimension
            elif image.ndim == 3 and image.shape[2] in [3, 4]:
                image = image.permute(2, 0, 1)
        else:
            # Ensure tensor is float
            image = image.float()

        # Move to next index, cycling back to start if needed
        self._current_index = (self._current_index + 1) % self.num_samples

        return image
