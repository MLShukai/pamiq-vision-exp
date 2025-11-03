import pytest
import torch

from exp.envs.image_env import ImageEnvironment
from exp.envs.transform import create_transform


class TestImageEnvironment:
    def test_observe_basic_functionality(self):
        """Test that observe method returns the generated image tensor."""
        # Create a simple generator that returns a constant image
        test_image = torch.rand(3, 64, 64)  # RGB image of size 64x64

        def generator():
            return test_image

        env = ImageEnvironment(generator)
        observed = env.observe()

        # Without transform, should return the original image
        assert observed.shape == test_image.shape
        assert torch.allclose(observed, test_image)

    def test_with_transform(self):
        """Test that transform is correctly applied to images."""
        test_image = torch.rand(3, 64, 64)

        def generator():
            return test_image

        transform = create_transform(size=(32, 32))
        env = ImageEnvironment(generator, transform=transform)
        observed = env.observe()

        # Transform should resize and standardize
        assert observed.shape == (3, 32, 32)
        assert torch.isclose(observed.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.isclose(observed.std(), torch.tensor(1.0), atol=0.1)

    def test_without_transform(self):
        """Test that images pass through unchanged without transform."""
        test_image = torch.ones(3, 64, 64) * 2.0  # Image with all values = 2

        def generator():
            return test_image

        env = ImageEnvironment(generator, transform=None)
        observed = env.observe()

        assert torch.allclose(observed, test_image)
        assert observed.mean() == 2.0

    def test_invalid_image_shape(self):
        """Test that an error is raised for invalid image shapes."""

        # Create generator that returns incorrectly shaped image (missing channel dimension)
        def invalid_generator():
            return torch.ones(64, 64)

        env = ImageEnvironment(invalid_generator)

        with pytest.raises(ValueError, match="Generated image must have 3 dimensions"):
            env.observe()
