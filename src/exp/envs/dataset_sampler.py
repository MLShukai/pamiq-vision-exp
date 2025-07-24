from collections.abc import Sized
from typing import override

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DatasetSelectOnlyImage[T](Dataset[T]):
    """Wrapper that extracts only images from datasets that return (image,
    label) tuples.

    Many torchvision datasets (e.g., CIFAR-10, CIFAR-100, ImageNet)
    return tuples of (image, label) when accessed. This wrapper extracts
    only the image component, making it compatible with components that
    expect just images.
    """

    def __init__(self, dataset: Dataset[tuple[T, int]]) -> None:
        """Initialize the image-only dataset wrapper.

        Args:
            dataset: A dataset that returns (image, label) tuples
        """
        self._dataset = dataset

    @override
    def __getitem__(self, index: int) -> T:
        return self._dataset[index][0]

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        raise RuntimeError(...)


class DatasetSampler:
    """Sequential sampler for torch datasets.

    This class wraps torchvision datasets (e.g., CIFAR-10, CIFAR-100,
    ImageNet) and provides sequential iteration over images. Images are
    accessed in their original order without shuffling.

    For shuffled access, wrap the dataset with ShuffleDataset before
    passing to this sampler.
    """

    def __init__(
        self,
        dataset: Dataset[Image.Image | torch.Tensor],
        *,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the dataset sampler.

        Args:
            dataset: A torchvision dataset instance
            max_samples: Maximum number of images to use. If None, use all images.
                        Must not exceed dataset size.

        Raises:
            ValueError: If max_samples exceeds the dataset size
            ValueError: If dataset doesn't implement __len__ method
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

        # Initialize current index for sequential iteration
        self._current_index = 0

    def __call__(self) -> torch.Tensor:
        """Get the next image in sequence as a torch tensor.

        Returns:
            Image tensor with shape [C, H, W] as float
        """
        # Get image from dataset (returns (image, label) tuple)
        image = self.dataset[self._current_index]

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


class ShuffleDataset[T](Dataset[T]):
    """Wrapper that shuffles the indices of a dataset.

    This class wraps any torch Dataset and provides shuffled access to its
    elements. The shuffling is deterministic based on the provided seed.

    Example:
        >>> dataset = torchvision.datasets.CIFAR10(...)
        >>> shuffled = ShuffleDataset(dataset, seed=42)
        >>> sampler = DatasetSampler(shuffled)  # Sequential access to shuffled data
    """

    def __init__(self, dataset: Dataset[T], seed: int | None = 8391) -> None:
        """Initialize the shuffled dataset wrapper.

        Args:
            dataset: The dataset to shuffle. Must implement __len__.
            seed: Random seed for reproducible shuffling.

        Raises:
            ValueError: If dataset doesn't implement __len__ method
        """
        super().__init__()
        if not isinstance(dataset, Sized):
            raise ValueError("Dataset must implement __len__ method")

        self._dataset = dataset

        self._indices = np.arange(len(dataset))
        rng = np.random.RandomState(seed)
        rng.shuffle(self._indices)

    @override
    def __getitem__(self, index: int) -> T:
        return self._dataset[self._indices[index]]

    def __len__(self) -> int:
        return len(self._dataset)
