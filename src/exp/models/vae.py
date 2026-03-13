"""Simple convolutional VAE for baseline comparison.

Follows the common encoder interface: input [B, C, T, H, W] -> output [B, latent_dim].
The VAE averages over time and encodes spatial frames into a flat latent vector.
"""

import math
from typing import override

import torch
import torch.nn as nn
from torch import Tensor

from exp.models.components.patchfier import VideoPatchifier
from exp.utils import size_3d, size_3d_to_tuple

# Spatial size of the feature map before flattening in the encoder
# (and after reshaping in the decoder). Determined by AdaptiveAvgPool2d.
_SPATIAL_POOL_SIZE = 4
_CHANNEL_MULTIPLIER = 4


class VAEEncoder(nn.Module):
    """VAE encoder following the common encoder interface.

    Collapses temporal frames via averaging, then encodes the resulting
    2D frame through convolutional layers into a flat latent vector.

    Input: [B, C, T, H, W] -> Output: [B, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self._latent_dim = latent_dim

        self._conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * _CHANNEL_MULTIPLIER, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(_SPATIAL_POOL_SIZE),
        )

        conv_out_dim = base_channels * _CHANNEL_MULTIPLIER * _SPATIAL_POOL_SIZE**2
        self._fc_mu = nn.Linear(conv_out_dim, latent_dim)
        self._fc_logvar = nn.Linear(conv_out_dim, latent_dim)

    def _encode_features(self, video: Tensor) -> Tensor:
        """Extract convolutional features from a video by time-averaging.

        Args:
            video: Input video [B, C, T, H, W].

        Returns:
            Flattened feature vector [B, conv_out_dim].
        """
        frame = video.mean(dim=2)  # [B, C, H, W]
        return self._conv(frame).flatten(1)

    @override
    def forward(self, video: Tensor) -> Tensor:
        """Encode video to latent representation.

        Args:
            video: Input video [B, C, T, H, W].

        Returns:
            Latent mean [B, latent_dim].
        """
        h = self._encode_features(video)
        return self._fc_mu(h)

    def encode_with_logvar(self, video: Tensor) -> tuple[Tensor, Tensor]:
        """Encode to (mu, log_var) for training with KL loss.

        Args:
            video: Input video [B, C, T, H, W].

        Returns:
            Tuple of (mu, log_var), each [B, latent_dim].
        """
        h = self._encode_features(video)
        return self._fc_mu(h), self._fc_logvar(h)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample from latent distribution using the reparameterization trick.

        Args:
            mu: Mean of the latent distribution [B, latent_dim].
            log_var: Log-variance of the latent distribution [B, latent_dim].

        Returns:
            Sampled latent vector [B, latent_dim].
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """VAE decoder that reconstructs spatial frames from latent vectors.

    Maps a flat latent vector back to a 2D image through transposed
    convolutions. Output spatial size is determined by the number of
    upsampling layers (3 layers of stride-2 from a 4x4 feature map ->
    32x32).
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self._base_channels = base_channels

        conv_out_dim = base_channels * _CHANNEL_MULTIPLIER * _SPATIAL_POOL_SIZE**2
        self._fc = nn.Linear(latent_dim, conv_out_dim)
        self._deconv = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * _CHANNEL_MULTIPLIER, base_channels * 2, 4, 2, 1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, in_channels, 4, 2, 1),
        )

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Decode latent vector to image.

        Args:
            z: Latent vector [B, latent_dim].

        Returns:
            Reconstructed image [B, C, 32, 32].
        """
        h = self._fc(z)
        h = h.reshape(
            -1,
            self._base_channels * _CHANNEL_MULTIPLIER,
            _SPATIAL_POOL_SIZE,
            _SPATIAL_POOL_SIZE,
        )
        return self._deconv(h)


def create_vae(
    video_shape: tuple[int, int, int] = (16, 224, 224),
    tubelet_size: size_3d = (2, 16, 16),
    in_channels: int = 3,
    embed_dim: int = 128,
    base_channels: int = 32,
    **kwargs: object,
) -> dict[str, nn.Module]:
    """Create a VAE encoder-decoder pair.

    Accepts video_shape/tubelet_size/embed_dim for Hydra config compatibility.
    Computes ``latent_dim = n_tubelets_total * embed_dim`` internally.
    The returned Encoder/Decoder themselves are tubelet-agnostic.

    Args:
        video_shape: Input video dimensions as (T, H, W).
        tubelet_size: Spatiotemporal patch size as (t, h, w) or scalar.
        in_channels: Number of input channels (e.g. 3 for RGB).
        embed_dim: Embedding dimension per tubelet.
        base_channels: Base channel count for conv layers.
        **kwargs: Ignored (accepts extra config keys gracefully).

    Returns:
        Dictionary with ``'encoder'`` and ``'decoder'`` modules.
    """
    tubelet_size = size_3d_to_tuple(tubelet_size)
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)
    latent_dim = math.prod(n_tubelets) * embed_dim

    encoder = VAEEncoder(
        latent_dim=latent_dim,
        in_channels=in_channels,
        base_channels=base_channels,
    )
    decoder = VAEDecoder(
        latent_dim=latent_dim,
        in_channels=in_channels,
        base_channels=base_channels,
    )
    return {
        "encoder": encoder,
        "decoder": decoder,
    }
