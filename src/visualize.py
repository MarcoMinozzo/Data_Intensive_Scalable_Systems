"""Quick visualisation helpers for paper figures (reconstructions, class counts)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.models.vae import ConvVAE


@torch.no_grad()
def plot_reconstructions(
    model: ConvVAE,
    images_batch: torch.Tensor,
    save_path: Optional[Path] = None,
    n: int = 8,
) -> Path:
    """Save a grid of original vs reconstructed galaxy images."""
    model.eval()
    device = next(model.parameters()).device
    x = images_batch[:n].to(device)
    recon, _, _ = model(x)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(x[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Real")
    axes[1, 0].set_ylabel("Recon")
    plt.suptitle("VAE reconstructions (Galaxy10)")
    plt.tight_layout()

    save_path = Path(save_path) if save_path else config.OUTPUT_FIGURES / "vae_reconstructions.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_class_counts(counts: dict, title: str, save_path: Optional[Path] = None) -> Path:
    """Bar chart of class counts for imbalance discussion."""
    ids = sorted(counts.keys())
    names = [config.CLASS_NAMES[i] for i in ids]
    values = [counts[i] for i in ids]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(ids)), values, color="#4C72B0")
    # Highlight minority classes used in the research question.
    for i, cid in enumerate(ids):
        if cid in config.MINORITY_CLASS_IDS:
            bars[i].set_color("#C44E52")
    plt.xticks(range(len(ids)), names, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()

    save_path = Path(save_path) if save_path else config.OUTPUT_FIGURES / "class_counts_train.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_per_class_f1_comparison(
    metrics_real: dict,
    metrics_aug: dict,
    save_path: Optional[Path] = None,
) -> Path:
    """Side-by-side per-class F1 for real-only vs real+VAE."""
    ids = list(range(config.NUM_CLASSES))
    names = [config.CLASS_NAMES[i] for i in ids]
    f1_real = [metrics_real["per_class"][str(i)]["f1"] for i in ids]
    f1_aug = [metrics_aug["per_class"][str(i)]["f1"] for i in ids]

    x = np.arange(len(ids))
    width = 0.38
    plt.figure(figsize=(11, 4))
    plt.bar(x - width / 2, f1_real, width, label="Real only")
    plt.bar(x + width / 2, f1_aug, width, label="Real + VAE")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("F1")
    plt.ylim(0, 1.05)
    plt.title("Per-class F1: real-only vs real+VAE synthetics")
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path) if save_path else config.OUTPUT_FIGURES / "f1_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
