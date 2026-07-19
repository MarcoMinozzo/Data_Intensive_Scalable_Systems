"""
Generate synthetic galaxy images with a trained VAE for class balancing.

Strategy:
  1) Encode real training images of a target class to estimate empirical mu.
  2) Sample z ~ N(class_mu, class_std) in latent space.
  3) Decode to RGB images in [0, 1] at VAE resolution (64x64).

Memory note:
  Synthetics stay at 64x64. Real images are referenced by index (no full copy).
  The CNN transform resizes both on the fly — avoids OOM from stacking 256x256
  duplicates of the training set.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config
from src.models.vae import ConvVAE


@torch.no_grad()
def estimate_class_latent_stats(
    model: ConvVAE,
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """For each class, estimate mean/std of encoder mu on the training split."""
    model.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE_VAE, config.IMAGE_SIZE_VAE)),
            transforms.ToTensor(),
        ]
    )
    stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for class_id in range(config.NUM_CLASSES):
        class_indices = train_idx[labels[train_idx] == class_id]
        mus: List[torch.Tensor] = []
        for start in range(0, len(class_indices), batch_size):
            chunk = class_indices[start : start + batch_size]
            batch = torch.stack(
                [tfm(Image.fromarray(images[i].astype(np.uint8))) for i in chunk]
            ).to(device)
            mu, _ = model.encode(batch)
            mus.append(mu.cpu())
        mus_t = torch.cat(mus, dim=0)
        stats[class_id] = (mus_t.mean(dim=0), mus_t.std(dim=0).clamp_min(1e-3))
        print(f"[gen] latent stats class={class_id} n={len(class_indices)}")
    return stats


@torch.no_grad()
def generate_class_images(
    model: ConvVAE,
    class_id: int,
    n: int,
    class_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Generate n RGB uint8 images (64, 64, 3) for a morphology class.
    """
    model.eval()
    mu, std = class_stats[class_id]
    mu = mu.to(device)
    std = std.to(device)

    out = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        z = mu + std * torch.randn(b, model.latent_dim, device=device)
        recon = model.decode(z).cpu()
        arr = (recon.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
        out.append(arr)
        remaining -= b
    return np.concatenate(out, axis=0)


class MixedRealSyntheticDataset(Dataset):
    """
    Training set = real Galaxy10 indices + compact 64x64 VAE synthetics.

    Avoids duplicating the full-resolution real training images in RAM.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        real_indices: np.ndarray,
        syn_images: Optional[np.ndarray],
        syn_labels: Optional[np.ndarray],
        transform=None,
    ):
        self.images = images
        self.labels = labels
        self.real_indices = np.asarray(real_indices)
        self.syn_images = syn_images if syn_images is not None else np.empty((0, 64, 64, 3), dtype=np.uint8)
        self.syn_labels = (
            syn_labels if syn_labels is not None else np.empty((0,), dtype=np.int64)
        )
        self.transform = transform
        self.n_real = len(self.real_indices)

    def __len__(self) -> int:
        return self.n_real + len(self.syn_labels)

    def __getitem__(self, i: int):
        if i < self.n_real:
            idx = self.real_indices[i]
            pil = Image.fromarray(self.images[idx].astype(np.uint8))
            label = int(self.labels[idx])
        else:
            j = i - self.n_real
            pil = Image.fromarray(self.syn_images[j].astype(np.uint8))
            label = int(self.syn_labels[j])

        x = self.transform(pil) if self.transform else transforms.ToTensor()(pil)
        return x, label


def build_balanced_synthetic_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    model: ConvVAE,
    device: torch.device,
    target_per_class: Optional[int] = config.SYNTHETIC_TARGET_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Generate only the synthetic 64x64 images needed to balance classes.
    Does not copy real images.
    """
    train_labels = labels[train_idx]
    counts = {c: int(np.sum(train_labels == c)) for c in range(config.NUM_CLASSES)}
    target = target_per_class or max(counts.values())

    print("[gen] Estimating class-conditional latent stats...")
    class_stats = estimate_class_latent_stats(model, images, labels, train_idx, device)

    syn_chunks: List[np.ndarray] = []
    syn_label_list: List[int] = []
    added = {c: 0 for c in range(config.NUM_CLASSES)}

    for class_id, count in counts.items():
        need = max(0, target - count)
        if need == 0:
            continue
        syn = generate_class_images(model, class_id, need, class_stats, device)
        syn_chunks.append(syn)
        syn_label_list.extend([class_id] * need)
        added[class_id] = need
        print(f"[gen] class={class_id} ({config.CLASS_NAMES[class_id]}): +{need} synthetics")

    if syn_chunks:
        syn_images = np.concatenate(syn_chunks, axis=0)
        syn_labels = np.asarray(syn_label_list, dtype=np.int64)
    else:
        syn_images = np.empty((0, config.IMAGE_SIZE_VAE, config.IMAGE_SIZE_VAE, 3), dtype=np.uint8)
        syn_labels = np.empty((0,), dtype=np.int64)

    print(f"[gen] total synthetics={len(syn_labels)} shape={syn_images.shape}")
    return syn_images, syn_labels, added


def make_mixed_loader(
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    syn_images: np.ndarray,
    syn_labels: np.ndarray,
    transform,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """DataLoader over real train indices + VAE synthetics."""
    ds = MixedRealSyntheticDataset(
        images=images,
        labels=labels,
        real_indices=train_idx,
        syn_images=syn_images,
        syn_labels=syn_labels,
        transform=transform,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# Backwards-compatible aliases used by older notebook cells.
def build_balanced_training_set(*args, **kwargs):
    """Deprecated alias — prefer build_balanced_synthetic_arrays + make_mixed_loader."""
    return build_balanced_synthetic_arrays(*args, **kwargs)


def make_loader_from_arrays(images, labels, transform, batch_size, shuffle):
    """Deprecated: kept for notebook compatibility with ArrayDataset."""
    from torch.utils.data import Dataset as _DS

    class ArrayDataset(_DS):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            pil = Image.fromarray(self.images[i].astype(np.uint8))
            x = self.transform(pil) if self.transform else transforms.ToTensor()(pil)
            return x, int(self.labels[i])

    return DataLoader(
        ArrayDataset(images, labels, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
