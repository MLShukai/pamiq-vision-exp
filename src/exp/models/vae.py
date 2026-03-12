"""Simple convolutional VAE for baseline comparison.

Follows the common encoder interface: input [B, C, T, H, W] -> output [B, n_tubelets, embed_dim].
The VAE averages over time and encodes spatial frames. Output is reshaped to match
V-JEPA's output format so that the same LightWeightDecoder can be used for evaluation.
"""

from typing import override

import torch
import torch.nn as nn
from torch import Tensor

from exp.models.components.patchfier import VideoPatchifier
from exp.utils import size_3d, size_3d_to_tuple


class VAEEncoder(nn.Module):
    """VAE encoder following the common encoder interface.

    Input: [B, C, T, H, W] -> Output: [B, n_tubelets_total, embed_dim]
    """

    def __init__(
        self,
        n_tubelets: tuple[int, int, int],
        embed_dim: int,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self._n_tubelets = n_tubelets
        self._embed_dim = embed_dim
        n_tubelets_total = n_tubelets[0] * n_tubelets[1] * n_tubelets[2]
        latent_dim = n_tubelets_total * embed_dim

        self._conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )

        enc_out_dim = base_channels * 4 * 4 * 4
        self._fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self._fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        self._n_tubelets_total = n_tubelets_total

    @override
    def forward(self, video: Tensor, masks: Tensor | None = None) -> Tensor:
        """Encode video to latent representation.

        Args:
            video: Input video [B, C, T, H, W].
            masks: Unused. Accepted for interface compatibility.

        Returns:
            Latent mean [B, n_tubelets_total, embed_dim].
        """
        frame = video.mean(dim=2)  # [B, C, H, W]
        h = self._conv(frame).flatten(1)
        mu = self._fc_mu(h)
        return mu.reshape(-1, self._n_tubelets_total, self._embed_dim)

    def encode_with_logvar(self, video: Tensor) -> tuple[Tensor, Tensor]:
        """Encode to (mu, log_var) for training with KL loss.

        Args:
            video: Input video [B, C, T, H, W].

        Returns:
            Tuple of (mu, log_var), each [B, latent_dim].
        """
        frame = video.mean(dim=2)
        h = self._conv(frame).flatten(1)
        return self._fc_mu(h), self._fc_logvar(h)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """VAE decoder that reconstructs spatial frames from latent vectors."""

    def __init__(
        self,
        n_tubelets: tuple[int, int, int],
        embed_dim: int,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self._base_channels = base_channels
        latent_dim = n_tubelets[0] * n_tubelets[1] * n_tubelets[2] * embed_dim
        enc_out_dim = base_channels * 4 * 4 * 4

        self._fc = nn.Linear(latent_dim, enc_out_dim)
        self._deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
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
        if z.dim() == 3:
            z = z.flatten(1)
        h = self._fc(z)
        h = h.reshape(-1, self._base_channels * 4, 4, 4)
        return self._deconv(h)


def create_vae(
    video_shape: tuple[int, int, int] = (16, 224, 224),
    tubelet_size: size_3d = (2, 16, 16),
    in_channels: int = 3,
    embed_dim: int = 128,
    base_channels: int = 32,
    **kwargs: object,
) -> dict[str, nn.Module]:
    """Create a VAE model set.

    Uses the same video_shape/tubelet_size/embed_dim as V-JEPA to ensure
    compatible output shapes for evaluation with LightWeightDecoder.

    Returns:
        Dictionary with 'encoder' and 'decoder'.
    """
    tubelet_size = size_3d_to_tuple(tubelet_size)
    n_tubelets = VideoPatchifier.compute_num_tubelets(video_shape, tubelet_size)

    encoder = VAEEncoder(
        n_tubelets=n_tubelets,
        embed_dim=embed_dim,
        in_channels=in_channels,
        base_channels=base_channels,
    )
    decoder = VAEDecoder(
        n_tubelets=n_tubelets,
        embed_dim=embed_dim,
        in_channels=in_channels,
        base_channels=base_channels,
    )
    return {
        "encoder": encoder,
        "decoder": decoder,
    }
