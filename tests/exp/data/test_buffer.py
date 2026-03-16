import pytest
import torch

from exp.data.buffer import FIFOReplayBuffer


class TestFIFOReplayBuffer:
    def test_add_increases_length(self):
        buf = FIFOReplayBuffer(max_size=10)
        assert len(buf) == 0

        buf.add(torch.randn(4))
        assert len(buf) == 1

        buf.add(torch.randn(4))
        assert len(buf) == 2

    def test_fifo_eviction(self):
        buf = FIFOReplayBuffer(max_size=3)

        # Add 4 items; the first should be evicted
        for i in range(4):
            buf.add(torch.full((2,), float(i)))

        assert len(buf) == 3
        # Verify storage contents directly: oldest (0) overwritten by newest (3)
        assert buf._storage is not None
        stored_values = {buf._storage[i, 0].item() for i in range(3)}
        assert 0.0 not in stored_values
        assert stored_values == {1.0, 2.0, 3.0}

    def test_save_and_load(self, tmp_path):
        buf = FIFOReplayBuffer(max_size=5)
        items = [torch.randn(3, 4) for _ in range(3)]
        for item in items:
            buf.add(item)

        save_dir = tmp_path / "buffer_state"
        buf.save(save_dir)

        loaded_buf = FIFOReplayBuffer(max_size=1)  # max_size will be overwritten
        loaded_buf.load(save_dir)

        assert len(loaded_buf) == len(buf)
        assert loaded_buf._max_size == buf._max_size
        assert loaded_buf._write_index == buf._write_index
        assert torch.equal(loaded_buf._storage[:3], buf._storage[:3])  # type: ignore[index]

    def test_get_data_returns_valid_items(self):
        buf = FIFOReplayBuffer(max_size=3)

        # Before full: should return only added items
        for i in range(2):
            buf.add(torch.full((2,), float(i)))
        data = buf.get_data()
        assert data.shape == (2, 2)
        assert data[0, 0].item() == 0.0
        assert data[1, 0].item() == 1.0

        # After full + eviction: should return max_size items
        for i in range(2, 5):
            buf.add(torch.full((2,), float(i)))
        data = buf.get_data()
        assert data.shape == (3, 2)

    def test_get_data_empty_raises(self):
        buf = FIFOReplayBuffer(max_size=5)
        with pytest.raises(RuntimeError):
            buf.get_data()

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            FIFOReplayBuffer(max_size=0)
        with pytest.raises(ValueError):
            FIFOReplayBuffer(max_size=-1)
