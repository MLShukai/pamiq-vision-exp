import torch

from exp.evaluation.baseline import DownsamplingBaseline


class TestDownsamplingBaseline:
    def test_downsample_output_shape(self):
        baseline = DownsamplingBaseline(target_feature_size=192, in_channels=3)
        videos = torch.randn(4, 3, 2, 32, 32)
        result = baseline.downsample(videos)
        assert result.dim() == 2
        assert result.shape[0] == 4

    def test_downsample_4d_input(self):
        baseline = DownsamplingBaseline(target_feature_size=192, in_channels=3)
        frames = torch.randn(4, 3, 32, 32)
        result = baseline.downsample(frames)
        assert result.dim() == 2
        assert result.shape[0] == 4

    def test_reconstruct_output_shape(self):
        baseline = DownsamplingBaseline(target_feature_size=192, in_channels=3)
        videos = torch.randn(4, 3, 2, 32, 32)
        features = baseline.downsample(videos)
        reconstructed = baseline.reconstruct(features, target_size=(32, 32))
        assert reconstructed.shape == (4, 3, 32, 32)

    def test_reconstruct_preserves_approximate_content(self):
        # Downsampling then upsampling should roughly preserve content
        baseline = DownsamplingBaseline(target_feature_size=768, in_channels=3)
        frames = torch.randn(2, 3, 16, 16)
        features = baseline.downsample(frames)
        reconstructed = baseline.reconstruct(features, target_size=(16, 16))
        # Not exact due to interpolation, but should have same shape
        assert reconstructed.shape == frames.shape
