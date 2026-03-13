import torch

from exp.evaluation.baseline import DownsamplingBaseline, create_downsampling_baseline


class TestDownsamplingBaseline:
    def test_forward_output_shape(self):
        feature_size = 1568 * 128  # = 200704
        baseline = DownsamplingBaseline(feature_size=feature_size, in_channels=3)
        video = torch.randn(2, 3, 16, 224, 224)
        result = baseline(video)
        assert result.shape == (2, feature_size)

    def test_forward_small_video(self):
        feature_size = 196 * 64  # = 12544
        baseline = DownsamplingBaseline(feature_size=feature_size, in_channels=3)
        video = torch.randn(4, 3, 8, 112, 112)
        result = baseline(video)
        assert result.shape == (4, feature_size)

    def test_no_learnable_parameters(self):
        baseline = DownsamplingBaseline(feature_size=1024)
        params = list(baseline.parameters())
        assert len(params) == 0


class TestCreateDownsamplingBaseline:
    def test_feature_size_from_config(self):
        baseline = create_downsampling_baseline(
            video_shape=(8, 112, 112), tubelet_size=(2, 16, 16), embed_dim=64
        )
        video = torch.randn(1, 3, 8, 112, 112)
        result = baseline(video)
        # n_tubelets = (4, 7, 7) -> 196, feature_size = 196 * 64 = 12544
        assert result.shape == (1, 12544)
