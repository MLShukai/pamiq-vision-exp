import pytest
import torch
from PIL import Image

from exp.envs.dataset_sampler import DatasetSampler, ShuffleDataset


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
        """Test that sampler iterates sequentially through indices."""

        def get_item(idx):
            return (torch.full((3, 32, 32), float(idx)), idx)

        mock_dataset.__getitem__.side_effect = get_item
        sampler = DatasetSampler(mock_dataset, max_samples=5)

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

        sampler = DatasetSampler(mock_dataset, max_samples=3)

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


class TestShuffleDataset:
    """Test the ShuffleDataset class."""

    @pytest.fixture
    def mock_dataset(self, mocker):
        """Create a mock dataset with sequential values."""
        dataset = mocker.MagicMock()
        dataset.__len__.return_value = 10

        # Return sequential values to test shuffling
        def get_item(idx):
            return idx

        dataset.__getitem__.side_effect = get_item
        return dataset

    def test_shuffling_with_seed(self, mock_dataset):
        """Test that ShuffleDataset properly shuffles indices."""
        # Create two datasets with same seed
        shuffle1 = ShuffleDataset(mock_dataset, seed=42)
        shuffle2 = ShuffleDataset(mock_dataset, seed=42)

        # Create dataset with different seed
        shuffle3 = ShuffleDataset(mock_dataset, seed=123)

        # Collect values
        values1 = [shuffle1[i] for i in range(10)]
        values2 = [shuffle2[i] for i in range(10)]
        values3 = [shuffle3[i] for i in range(10)]

        # Same seed should produce same order
        assert values1 == values2

        # Different seed should produce different order
        assert values1 != values3

        # Should contain all original values
        assert sorted(values1) == list(range(10))
        assert sorted(values3) == list(range(10))

    def test_len_method(self, mock_dataset):
        """Test that __len__ returns correct dataset length."""
        shuffle_dataset = ShuffleDataset(mock_dataset, seed=42)
        assert len(shuffle_dataset) == 10

    def test_dataset_not_sized(self, mocker):
        """Test that ValueError is raised when dataset doesn't implement
        __len__."""
        mock_dataset = mocker.Mock()

        with pytest.raises(ValueError, match="Dataset must implement __len__ method"):
            ShuffleDataset(mock_dataset)

    def test_getitem_returns_correct_values(self, mock_dataset):
        """Test that getitem returns the correct shuffled values."""
        shuffle_dataset = ShuffleDataset(mock_dataset, seed=42)

        # Get all values
        values = [shuffle_dataset[i] for i in range(10)]

        # Should have all values from 0 to 9
        assert set(values) == set(range(10))

        # Values should not be in sequential order (with high probability)
        assert values != list(range(10))

    def test_default_seed(self, mock_dataset):
        """Test that default seed produces consistent results."""
        # Create datasets without specifying seed (uses default seed=8391)
        shuffle1 = ShuffleDataset(mock_dataset)
        shuffle2 = ShuffleDataset(mock_dataset)

        # Should produce same order with default seed
        values1 = [shuffle1[i] for i in range(10)]
        values2 = [shuffle2[i] for i in range(10)]

        assert values1 == values2
