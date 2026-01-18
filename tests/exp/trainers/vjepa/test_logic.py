import pytest
import torch

from exp.models.vjepa import Encoder, Predictor
from exp.trainers.vjepa.logic import VJEPATrainingLogic


@pytest.fixture
def models():
    """Create small encoder and predictor models for testing."""
    n_tubelets = (2, 4, 4)
    hidden_dim = 32
    embed_dim = 16

    context_encoder = Encoder(
        patchifier=None,
        n_tubelets=n_tubelets,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        depth=1,
        num_heads=2,
    )
    target_encoder = context_encoder.clone()
    predictor = Predictor(
        n_tubelets=n_tubelets,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        depth=1,
        num_heads=2,
    )
    return context_encoder, target_encoder, predictor


@pytest.fixture
def training_logic(models):
    """Create VJEPATrainingLogic instance for testing."""
    context_encoder, target_encoder, predictor = models
    optimizer = torch.optim.Adam(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3,
    )
    return VJEPATrainingLogic(
        context_encoder=context_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        optimizer=optimizer,
        ema_momentum=0.99,
    )


@pytest.fixture
def batch():
    """Create a batch of test data."""
    batch_size = 2
    n_tubelets = 2 * 4 * 4
    hidden_dim = 32

    videos = torch.randn(batch_size, n_tubelets, hidden_dim)
    masks_for_context_encoder = torch.zeros(batch_size, n_tubelets, dtype=torch.bool)
    masks_for_context_encoder[:, :8] = True
    targets_for_predictor = torch.zeros(batch_size, n_tubelets, dtype=torch.bool)
    targets_for_predictor[:, :8] = True

    return videos, masks_for_context_encoder, targets_for_predictor


class TestVJEPATrainingLogic:
    def test_compute_loss(self, training_logic, batch):
        videos, masks, targets = batch
        result = training_logic.compute_loss(videos, masks, targets)

        assert torch.isfinite(result.loss)
        assert result.loss_per_data.shape == (videos.shape[0],)
        assert torch.all(torch.isfinite(result.loss_per_data))

    def test_train_step(self, training_logic, batch):
        videos, masks, targets = batch
        result = training_logic.train_step(videos, masks, targets)

        assert torch.isfinite(result.loss)
