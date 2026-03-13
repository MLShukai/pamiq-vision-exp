import torch

from exp.models.vae import VAEDecoder, VAEEncoder
from exp.trainers.vae.logic import VAETrainingLogic

# Small dimensions for fast test execution
_LATENT_DIM = 64  # 2*2*2 tubelets * 8 embed_dim
_BASE_CHANNELS = 8


def _make_logic(**overrides: object) -> VAETrainingLogic:
    """Build a minimal VAETrainingLogic for testing."""
    encoder = VAEEncoder(
        latent_dim=_LATENT_DIM, in_channels=3, base_channels=_BASE_CHANNELS
    )
    decoder = VAEDecoder(
        latent_dim=_LATENT_DIM, in_channels=3, base_channels=_BASE_CHANNELS
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    kwargs: dict[str, object] = dict(
        encoder=encoder, decoder=decoder, optimizer=optimizer
    )
    kwargs.update(overrides)
    return VAETrainingLogic(**kwargs)  # type: ignore[arg-type]


class TestVAETrainingLogic:
    def test_train_step_from_batch(self):
        logic = _make_logic(kl_weight=1e-3)
        batch = torch.randn(2, 3, 4, 32, 32)

        result = logic.train_step_from_batch(batch)

        assert torch.isfinite(result.loss)
        assert "recon_loss" in result.metrics
        assert "kl_loss" in result.metrics

    def test_parameters_update(self):
        logic = _make_logic()
        encoder = logic._encoder

        params_before = [p.clone() for p in encoder.parameters()]
        batch = torch.randn(2, 3, 4, 32, 32)
        logic.train_step_from_batch(batch)

        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, list(encoder.parameters()))
        )
        assert changed
