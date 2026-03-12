import torch.nn.functional as F
from torch import Tensor


class DownsamplingBaseline:
    """Baseline that uses simple downsampling for comparison.

    Downsamples video frames to match the byte size of encoder output,
    providing a fair comparison point.
    """

    def __init__(
        self,
        target_feature_size: int,
        in_channels: int = 3,
    ) -> None:
        """Initialize downsampling baseline.

        Args:
            target_feature_size: Total number of feature values to match
                (n_tubelets * embed_dim from encoder).
            in_channels: Number of input channels.
        """
        self._target_feature_size = target_feature_size
        self._in_channels = in_channels
        # Compute spatial size for downsampled representation
        # target_feature_size = C * pixels_per_frame
        pixels_per_channel = target_feature_size // in_channels
        self._target_spatial = max(round(pixels_per_channel**0.5), 1)

    def downsample(self, videos: Tensor) -> Tensor:
        """Downsample videos to match target byte size.

        Args:
            videos: Input videos [N, C, T, H, W] or frames [N, C, H, W]

        Returns:
            Downsampled representation [N, target_feature_size]
        """
        if videos.dim() == 5:
            # Average over time, then resize spatial
            frames = videos.mean(dim=2)  # [N, C, H, W]
        else:
            frames = videos

        downsampled = F.interpolate(
            frames,
            size=(self._target_spatial, self._target_spatial),
            mode="bilinear",
            align_corners=False,
        )

        return downsampled.flatten(1)  # [N, C * target_spatial^2]

    def reconstruct(self, features: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Reconstruct by upsampling back to original spatial size.

        Args:
            features: Downsampled features [N, feat_dim]
            target_size: Target spatial size (H, W)

        Returns:
            Reconstructed frames [N, C, H, W]
        """
        n = features.shape[0]
        spatial = self._target_spatial
        frames = features.reshape(n, self._in_channels, spatial, spatial)

        return F.interpolate(
            frames,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
