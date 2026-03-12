from pathlib import Path

import pytest
import torch
import torch.nn as nn

from exp.training.checkpoint import CheckpointManager


def test_save_creates_checkpoint_directory(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "test_exp")
    model = nn.Linear(4, 2)

    path = manager.save({"model": model}, elapsed_seconds=5.0)

    assert path.exists()
    assert path.is_dir()
    assert path.name == "5.ckpt"


def test_save_creates_model_files(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "test_exp")
    model_a = nn.Linear(4, 2)
    model_b = nn.Linear(2, 3)

    path = manager.save({"encoder": model_a, "decoder": model_b}, elapsed_seconds=10.0)

    assert (path / "encoder.pt").exists()
    assert (path / "decoder.pt").exists()


def test_save_and_load_preserves_parameters(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "test_exp")
    model = nn.Linear(4, 2)
    original_weight = model.weight.data.clone()

    path = manager.save({"model": model}, elapsed_seconds=10.0)

    # Modify model
    nn.init.zeros_(model.weight)
    assert not torch.equal(model.weight.data, original_weight)

    # Load and verify
    CheckpointManager.load(path, {"model": model})
    assert torch.equal(model.weight.data, original_weight)


def test_load_missing_file_raises(tmp_path: Path):
    nonexistent_path = tmp_path / "nonexistent.ckpt"
    nonexistent_path.mkdir(parents=True)
    model = nn.Linear(4, 2)

    with pytest.raises(FileNotFoundError):
        CheckpointManager.load(nonexistent_path, {"model": model})


def test_load_encoder_returns_eval_mode(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "test_exp")
    encoder = nn.Linear(4, 2)
    encoder.train()

    path = manager.save({"encoder": encoder}, elapsed_seconds=15.0)

    loaded_encoder = nn.Linear(4, 2)
    loaded_encoder.train()
    result = CheckpointManager.load_encoder(path, "encoder", loaded_encoder)

    assert not result.training
    assert result is loaded_encoder


def test_save_with_extra_data(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "test_exp")
    model = nn.Linear(4, 2)
    extra = {"learning_rate": 0.001, "epoch": 5}

    path = manager.save({"model": model}, elapsed_seconds=20.0, extra=extra)

    assert (path / "extra.pt").exists()
    loaded_extra = torch.load(path / "extra.pt", weights_only=False)
    assert loaded_extra["learning_rate"] == 0.001
    assert loaded_extra["epoch"] == 5


def test_multiple_checkpoints(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "test_exp")
    model = nn.Linear(4, 2)

    path1 = manager.save({"model": model}, elapsed_seconds=10.0)
    path2 = manager.save({"model": model}, elapsed_seconds=20.0)
    path3 = manager.save({"model": model}, elapsed_seconds=30.0)

    assert path1 != path2 != path3
    assert path1.name == "10.ckpt"
    assert path2.name == "20.ckpt"
    assert path3.name == "30.ckpt"
    assert all(p.exists() for p in [path1, path2, path3])
