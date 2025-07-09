import pytest
import torch

from exp.envs.image_env import ImageEnvironment


class TestImageEnvironment:
    def test_observe_basic_functionality(self):
        """Test that observe method returns the correct image tensor."""
        # Create a simple generator that returns a constant image
        test_image = torch.rand(3, 64, 64)  # RGB image of size 64x64

        def generator():
            return test_image

        env = ImageEnvironment(generator)
        observed = env.observe()

        # Since standardization is on by default, check that values are standardized
        assert observed.shape == test_image.shape
        assert torch.isclose(observed.mean(), torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(observed.std(), torch.tensor(1.0), atol=1e-5)

    def test_resize_functionality(self):
        """Test that size parameter correctly resizes images."""
        test_image = torch.ones(3, 64, 64)

        def generator():
            return test_image

        target_size = (32, 32)

        env = ImageEnvironment(generator, size=target_size)
        observed = env.observe()

        assert observed.shape == (3, 32, 32)

    @pytest.mark.parametrize("standardize,expected_mean", [(True, 0.0), (False, 2.0)])
    def test_standardize_parameter(self, standardize, expected_mean):
        """Test that standardize parameter works correctly."""
        # Create image with known mean and std
        test_image = torch.ones(3, 64, 64) * 2.0  # Image with all values = 2

        def generator():
            return test_image

        env = ImageEnvironment(generator, standardize=standardize)
        observed = env.observe()

        assert torch.isclose(observed.mean(), torch.tensor(expected_mean), atol=1e-5)

    def test_dtype_parameter(self):
        """Test that dtype parameter works correctly."""
        test_image = torch.ones(3, 64, 64, dtype=torch.uint8) * 255

        def generator():
            return test_image

        env = ImageEnvironment(generator, dtype=torch.float16, standardize=False)
        observed = env.observe()

        assert observed.dtype == torch.float16
        assert torch.isclose(
            observed[0, 0, 0], torch.tensor(255.0, dtype=torch.float16)
        )

    def test_invalid_image_shape(self):
        """Test that an error is raised for invalid image shapes."""

        # Create generator that returns incorrectly shaped image (missing channel dimension)
        def invalid_generator():
            return torch.ones(64, 64)

        env = ImageEnvironment(invalid_generator)

        with pytest.raises(ValueError, match="Generated image must have 3 dimensions"):
            env.observe()
