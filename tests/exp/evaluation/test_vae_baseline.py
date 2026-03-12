import torch

from exp.evaluation.vae_baseline import VAEBaselineEvaluator


class TestVAEBaselineEvaluator:
    def test_train_returns_losses(self):
        evaluator = VAEBaselineEvaluator(in_channels=3, latent_dim=32)
        videos = torch.randn(8, 3, 4, 16, 16)
        losses = evaluator.train_vae(videos, num_epochs=3, batch_size=4)
        assert len(losses) == 3
        assert all(loss > 0 for loss in losses)

    def test_evaluate_returns_result(self):
        evaluator = VAEBaselineEvaluator(in_channels=3, latent_dim=32)
        videos = torch.randn(8, 3, 4, 16, 16)
        evaluator.train_vae(videos, num_epochs=2, batch_size=4)
        result = evaluator.evaluate(videos)
        assert result.recon_mae >= 0
        assert result.recon_mse >= 0
