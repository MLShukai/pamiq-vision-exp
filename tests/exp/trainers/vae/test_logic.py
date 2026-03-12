import torch

from exp.models.vae import VAEDecoder, VAEEncoder
from exp.trainers.vae.logic import VAETrainingLogic


class TestVAETrainingLogic:
    def test_train_step_from_batch(self):
        n_tubelets = (2, 2, 2)
        embed_dim = 8
        encoder = VAEEncoder(
            n_tubelets=n_tubelets, embed_dim=embed_dim, in_channels=3, base_channels=8
        )
        decoder = VAEDecoder(
            n_tubelets=n_tubelets, embed_dim=embed_dim, in_channels=3, base_channels=8
        )
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        logic = VAETrainingLogic(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            kl_weight=1e-3,
        )

        batch = torch.randn(2, 3, 4, 32, 32)
        result = logic.train_step_from_batch(batch)

        assert torch.isfinite(result.loss)
        assert "recon_loss" in result.metrics
        assert "kl_loss" in result.metrics

    def test_parameters_update(self):
        n_tubelets = (2, 2, 2)
        embed_dim = 8
        encoder = VAEEncoder(
            n_tubelets=n_tubelets, embed_dim=embed_dim, in_channels=3, base_channels=8
        )
        decoder = VAEDecoder(
            n_tubelets=n_tubelets, embed_dim=embed_dim, in_channels=3, base_channels=8
        )
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        logic = VAETrainingLogic(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
        )

        params_before = [p.clone() for p in encoder.parameters()]
        batch = torch.randn(2, 3, 4, 32, 32)
        logic.train_step_from_batch(batch)
        params_after = list(encoder.parameters())

        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert changed
