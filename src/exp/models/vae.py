"""Simple convolutional VAE for baseline comparison."""

from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class SimpleVAE(nn.Module):
    """Simple convolutional VAE for baseline reconstruction evaluation.

    Encodes video frames (averaged over time) to a latent space and
    reconstructs them. Used as a baseline to compare against V-JEPA
    representations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self._latent_dim = latent_dim

        # Encoder: conv layers with adaptive pooling
        self._encoder = nn.Sequential(
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

        # Decoder
        self._fc_decode = nn.Linear(latent_dim, enc_out_dim)
        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, in_channels, 4, 2, 1),
        )

        self._base_channels = base_channels

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input images [N, C, H, W].

        Returns:
            Tuple of (mu, log_var), each [N, latent_dim].
        """
        h = self._encoder(x)
        h = h.flatten(1)
        return self._fc_mu(h), self._fc_logvar(h)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to image.

        Args:
            z: Latent vector [N, latent_dim].

        Returns:
            Reconstructed image [N, C, 32, 32].
        """
        h = self._fc_decode(z)
        h = h.reshape(-1, self._base_channels * 4, 4, 4)
        return self._decoder(h)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input images [N, C, H, W].

        Returns:
            Tuple of (reconstructed, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
