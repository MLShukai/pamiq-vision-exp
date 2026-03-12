import torch
import torch.nn as nn

from exp.evaluation.reconstruction import ReconstructionEvaluator, ReconstructionResult


class SimpleEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> flatten to [B, 1, C*T*H*W]
        b = x.shape[0]
        return x.reshape(b, 1, -1)[:, :, :8]  # just take first 8 features


class SimpleDecoder(nn.Module):
    def __init__(self, feat_dim: int, out_shape: tuple[int, int, int, int]) -> None:
        super().__init__()
        self._out_shape = out_shape
        c, t, h, w = out_shape
        self._proj = nn.Linear(feat_dim, c * t * h * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.reshape(b, -1)
        x = self._proj(x)
        return x.reshape(b, *self._out_shape)


class TestReconstructionEvaluator:
    def test_encode_dataset_shape(self):
        encoder = SimpleEncoder()
        decoder = SimpleDecoder(8, (3, 2, 4, 4))
        evaluator = ReconstructionEvaluator(encoder, decoder)
        videos = torch.randn(10, 3, 2, 4, 4)
        features = evaluator.encode_dataset(videos, batch_size=4)
        assert features.shape == (10, 1, 8)

    def test_train_decoder_returns_losses(self):
        encoder = SimpleEncoder()
        decoder = SimpleDecoder(8, (3, 2, 4, 4))
        evaluator = ReconstructionEvaluator(encoder, decoder)
        videos = torch.randn(10, 3, 2, 4, 4)
        features = evaluator.encode_dataset(videos)
        losses = evaluator.train_decoder(features, videos, num_epochs=3, batch_size=5)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_evaluate_returns_result(self):
        encoder = SimpleEncoder()
        decoder = SimpleDecoder(8, (3, 2, 4, 4))
        evaluator = ReconstructionEvaluator(encoder, decoder)
        videos = torch.randn(10, 3, 2, 4, 4)
        features = evaluator.encode_dataset(videos)
        result = evaluator.evaluate(features, videos)
        assert isinstance(result, ReconstructionResult)
        assert result.mae >= 0
        assert result.mse >= 0

    def test_evaluate_identical_reconstruction(self):
        # If decoder outputs exactly the target, errors should be ~0
        encoder = nn.Identity()
        decoder = nn.Identity()
        evaluator = ReconstructionEvaluator(encoder, decoder)
        videos = torch.randn(5, 3, 2, 4, 4)
        # features = videos (identity encoder)
        result = evaluator.evaluate(videos, videos)
        assert result.mae < 1e-5
        assert result.mse < 1e-5
