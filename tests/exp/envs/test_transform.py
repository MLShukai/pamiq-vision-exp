import pytest
import torch
from torch import Tensor

from exp.envs.transform import create_transform, standardize


class TestStandardize:
    def test_standardize_simple(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize(tensor)

        assert torch.allclose(result.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(result.std(), torch.tensor(1.0), atol=1e-6)

    def test_standardize_2d(self):
        tensor = torch.randn(10, 10)
        result = standardize(tensor)

        assert torch.allclose(result.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(result.std(), torch.tensor(1.0), atol=1e-6)


class TestCreateTransform:
    def test_create_transform_basic(self):
        transform = create_transform(size=(224, 224))
        assert callable(transform)

    def test_create_transform_with_device(self):
        device = torch.device("cpu")
        transform = create_transform(size=(224, 224), device=device)
        assert callable(transform)

    def test_create_transform_with_dtype(self):
        dtype = torch.float32
        transform = create_transform(size=(224, 224), dtype=dtype)
        assert callable(transform)

    def test_transform_output_shape(self):
        import numpy as np
        from PIL import Image

        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

        transform = create_transform(size=(224, 224))
        result = transform(img)

        assert isinstance(result, Tensor)
        assert result.shape == (3, 224, 224)

    def test_transform_output_normalized(self):
        import numpy as np
        from PIL import Image

        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

        transform = create_transform(size=(224, 224))
        result = transform(img)

        # Check that output is approximately standardized
        assert torch.allclose(result.mean(), torch.tensor(0.0), atol=0.1)
        assert torch.allclose(result.std(), torch.tensor(1.0), atol=0.1)
