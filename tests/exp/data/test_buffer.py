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

    def test_sample_shape(self):
        buf = FIFOReplayBuffer(max_size=10)
        item_shape = (3, 4, 4)

        for _ in range(5):
            buf.add(torch.randn(*item_shape))

        batch = buf.sample(batch_size=3)
        assert batch.shape == (3, *item_shape)

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

    def test_sample_empty_raises(self):
        buf = FIFOReplayBuffer(max_size=5)
        with pytest.raises(RuntimeError):
            buf.sample(1)

    def test_sample_batch_too_large_raises(self):
        buf = FIFOReplayBuffer(max_size=10)
        buf.add(torch.randn(4))
        with pytest.raises(ValueError):
            buf.sample(2)

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

    def test_is_full(self):
        buf = FIFOReplayBuffer(max_size=2)
        assert not buf.is_full
        buf.add(torch.randn(4))
        assert not buf.is_full
        buf.add(torch.randn(4))
        assert buf.is_full

    def test_invalid_max_size(self):
        with pytest.raises(ValueError):
            FIFOReplayBuffer(max_size=0)
        with pytest.raises(ValueError):
            FIFOReplayBuffer(max_size=-1)
