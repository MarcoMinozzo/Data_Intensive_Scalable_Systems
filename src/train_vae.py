"""Training loop for the convolutional VAE."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.models.vae import ConvVAE, vae_loss


def train_vae(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = config.VAE_EPOCHS,
    latent_dim: int = config.VAE_LATENT_DIM,
    lr: float = config.VAE_LR,
    beta: float = config.VAE_BETA,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
) -> ConvVAE:
    """
    Train ConvVAE and optionally save the best validation checkpoint.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    save_path = Path(save_path) if save_path else config.OUTPUT_MODELS / "vae_galaxy10.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, _ in tqdm(train_loader, desc=f"VAE epoch {epoch}/{epochs}", leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            v_running, v_batches = 0.0, 0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    recon, mu, logvar = model(x)
                    loss, _, _ = vae_loss(recon, x, mu, logvar, beta=beta)
                    v_running += loss.item()
                    v_batches += 1
            val_loss = v_running / max(v_batches, 1)
            history["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "latent_dim": latent_dim,
                        "history": history,
                    },
                    save_path,
                )
        print(f"[VAE] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # If no validation set was provided, save final weights.
    if val_loader is None:
        torch.save(
            {"model_state": model.state_dict(), "latent_dim": latent_dim, "history": history},
            save_path,
        )

    # Reload best checkpoint when available.
    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    print(f"[VAE] Saved checkpoint: {save_path}")
    return model


def load_vae(path: Optional[Path] = None, device: Optional[torch.device] = None) -> ConvVAE:
    """Load a trained ConvVAE checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path) if path else config.OUTPUT_MODELS / "vae_galaxy10.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConvVAE(latent_dim=ckpt.get("latent_dim", config.VAE_LATENT_DIM)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
