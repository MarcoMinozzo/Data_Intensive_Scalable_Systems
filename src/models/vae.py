"""
Convolutional Variational Autoencoder (VAE).

Aligned with module labs (Week 6 / Implementing VAE Loss):
  - encode -> mu, logvar
  - reparameterize(mu, logvar)
  - decode(z)
  - loss = reconstruction + KL divergence

Adaptation for Galaxy10:
  Fully-connected MNIST VAEs from the labs are impractical for 256x256 RGB
  images, so we use a convolutional encoder/decoder while keeping the same
  probabilistic VAE formulation taught in class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Convolutional VAE for RGB galaxy images in [0, 1]."""

    def __init__(self, latent_dim: int = 32, in_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (B, 3, 64, 64) -> feature map -> mu/logvar
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: z -> feature map -> (B, 3, 64, 64)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),  # pixel values in [0, 1], as in the labs
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Same trick as in the module VAE labs.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """
    VAE loss = reconstruction + beta * KL, matching lab formulation.

    Labs use binary cross-entropy + closed-form KL between q(z|x) and N(0, I).
    """
    # Sum over pixels then mean over batch for stable reporting.
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_div, recon_loss.detach(), kl_div.detach()
