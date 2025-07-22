import pytest
import torch
from PIL import Image

from exp.envs.dataset_sampler import DatasetSampler


class TestDatasetSampler:
    """Test the DatasetSampler class."""

    @pytest.fixture
    def mock_dataset(self, mocker):
        """Create a mock dataset with 100 samples."""
        dataset = mocker.MagicMock()
        dataset.__len__.return_value = 100
        return dataset

    @pytest.fixture
    def mock_dataset_with_tensor(self, mocker):
        """Create a mock dataset that returns tensors."""
        dataset = mocker.MagicMock()
        dataset.__len__.return_value = 10
        dataset.__getitem__.return_value = (torch.randn(3, 32, 32), 0)
        return dataset

    @pytest.fixture
    def mock_dataset_with_pil(self, mocker):
        """Create a mock dataset that returns PIL images."""
        dataset = mocker.MagicMock()
        dataset.__len__.return_value = 10
        dataset.__getitem__.return_value = (Image.new("RGB", (32, 32), color="red"), 0)
        return dataset

    @pytest.mark.parametrize("max_samples", [1, 10, 50, 100])
    def test_initialization_with_max_samples(self, mock_dataset, max_samples):
        """Test initialization with various max_samples values."""
        sampler = DatasetSampler(mock_dataset, max_samples=max_samples)

        assert sampler.num_samples == max_samples

    def test_max_samples_exceeds_dataset_size(self, mock_dataset):
        """Test that ValueError is raised when max_samples > dataset size."""
        with pytest.raises(ValueError, match="max_samples .* exceeds dataset size"):
            DatasetSampler(mock_dataset, max_samples=150)

    def test_dataset_not_sized(self, mocker):
        """Test that ValueError is raised when dataset doesn't implement
        __len__."""
        mock_dataset = mocker.Mock()

        with pytest.raises(ValueError, match="Dataset must implement __len__ method"):
            DatasetSampler(mock_dataset)

    def test_indices_are_shuffled(self, mock_dataset):
        """Test that indices are properly shuffled."""

        # Create mock that returns unique values for each index
        def get_item(idx):
            return (torch.tensor([float(idx)]), idx)

        mock_dataset.__getitem__.side_effect = get_item

        sampler1 = DatasetSampler(mock_dataset, shuffle=True, seed=42, max_samples=10)
        sampler2 = DatasetSampler(mock_dataset, shuffle=True, seed=123, max_samples=10)
        sampler3 = DatasetSampler(mock_dataset, shuffle=True, seed=42, max_samples=10)

        # Collect sequence of values from each sampler
        seq1 = [sampler1()[0].item() for _ in range(10)]
        seq2 = [sampler2()[0].item() for _ in range(10)]
        seq3 = [sampler3()[0].item() for _ in range(10)]

        # Different seeds should produce different orderings
        assert seq1 != seq2

        # Same seed should produce same ordering
        assert seq1 == seq3

    @pytest.mark.parametrize(
        "image_data,expected_shape",
        [
            (Image.new("RGB", (32, 32), color="red"), (3, 32, 32)),
            (Image.new("L", (28, 28), color=128), (1, 28, 28)),
            (torch.randn(3, 64, 64), (3, 64, 64)),
        ],
    )
    def test_call_returns_correct_tensor(
        self, mock_dataset, image_data, expected_shape
    ):
        """Test that __call__ returns correct tensor format for different input
        types."""
        mock_dataset.__getitem__.return_value = (image_data, 0)
        sampler = DatasetSampler(mock_dataset, max_samples=5)

        result = sampler()

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == expected_shape

    def test_sequential_iteration(self, mock_dataset):
        """Test that sampler iterates sequentially through selected indices."""

        def get_item(idx):
            return (torch.full((3, 32, 32), float(idx)), idx)

        mock_dataset.__getitem__.side_effect = get_item
        sampler = DatasetSampler(mock_dataset, seed=42, max_samples=5)

        # First iteration: collect values
        first_iteration = []
        for _ in range(5):
            image = sampler()
            value = image[0, 0, 0].item()  # Get the fill value
            first_iteration.append(value)

        # Second iteration: should get same sequence
        second_iteration = []
        for _ in range(5):
            image = sampler()
            value = image[0, 0, 0].item()
            second_iteration.append(value)

        assert first_iteration == second_iteration

    def test_cycling_behavior(self, mock_dataset):
        """Test that sampler cycles back to beginning after reaching end."""

        # Create dataset that returns unique tensors for each index
        def get_item(idx):
            return (torch.tensor([float(idx)]), idx)

        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = get_item

        sampler = DatasetSampler(mock_dataset, max_samples=3, shuffle=False)

        # Collect values for two full cycles
        values = []
        for _ in range(6):
            tensor = sampler()
            values.append(tensor[0].item())

        # Should see the same pattern twice: 0,1,2,0,1,2
        assert values[:3] == values[3:]

    @pytest.mark.parametrize(
        "max_samples,expected",
        [
            (1, 1),
            (10, 10),
            (100, 100),
            (500, 500),
            (1000, 1000),
            (None, 1000),
        ],
    )
    def test_different_max_samples(self, mocker, max_samples, expected):
        """Test with various max_samples values."""
        mock_dataset = mocker.MagicMock()
        mock_dataset.__len__.return_value = 1000
        mock_dataset.__getitem__.return_value = (torch.randn(3, 32, 32), 0)

        sampler = DatasetSampler(mock_dataset, max_samples=max_samples)
        assert sampler.num_samples == expected
        # Verify by calling the sampler expected times
        # If it doesn't raise an error, the number of samples is correct
        for _ in range(expected):
            result = sampler()
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 32, 32)
