import torch

from exp.evaluation.baseline import DownsamplingBaseline


class TestDownsamplingBaseline:
    def test_forward_output_shape(self):
        baseline = DownsamplingBaseline(
            video_shape=(16, 224, 224),
            tubelet_size=(2, 16, 16),
            embed_dim=128,
            in_channels=3,
        )
        # n_tubelets = (8, 14, 14) -> total = 1568
        video = torch.randn(2, 3, 16, 224, 224)
        result = baseline(video)
        assert result.shape == (2, 1568, 128)

    def test_forward_small_video(self):
        baseline = DownsamplingBaseline(
            video_shape=(8, 112, 112),
            tubelet_size=(2, 16, 16),
            embed_dim=64,
            in_channels=3,
        )
        # n_tubelets = (4, 7, 7) -> total = 196
        video = torch.randn(4, 3, 8, 112, 112)
        result = baseline(video)
        assert result.shape == (4, 196, 64)

    def test_forward_ignores_masks(self):
        baseline = DownsamplingBaseline(
            video_shape=(8, 112, 112),
            tubelet_size=(2, 16, 16),
            embed_dim=64,
            in_channels=3,
        )
        video = torch.randn(2, 3, 8, 112, 112)
        masks = torch.ones(2, 196, dtype=torch.bool)
        result_no_mask = baseline(video)
        result_with_mask = baseline(video, masks)
        assert torch.equal(result_no_mask, result_with_mask)

    def test_no_learnable_parameters(self):
        baseline = DownsamplingBaseline(
            video_shape=(8, 112, 112),
            tubelet_size=(2, 16, 16),
            embed_dim=64,
        )
        params = list(baseline.parameters())
        assert len(params) == 0
