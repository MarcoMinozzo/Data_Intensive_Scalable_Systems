"""
Data download and loading utilities for Galaxy10 DECaLS.

Dataset citation / source (required for reproducibility):
  - Galaxy10 DECaLS: https://astronn.readthedocs.io/en/latest/galaxy10.html
  - GitHub: https://github.com/henrysky/Galaxy10
  - Labels from Galaxy Zoo; images from DESI Legacy Imaging Surveys.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config


def ensure_dirs() -> None:
    """Create project data/output directories if missing."""
    for path in [
        config.DATA_RAW,
        config.DATA_PROCESSED,
        config.OUTPUT_MODELS,
        config.OUTPUT_FIGURES,
        config.OUTPUT_METRICS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def download_galaxy10(destination: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download Galaxy10_DECals.h5 if it is not already present.

    Parameters
    ----------
    destination : Path, optional
        Target file path. Defaults to config.GALAXY10_H5_PATH.
    force : bool
        If True, re-download even when the file exists.
    """
    ensure_dirs()
    dest = Path(destination) if destination is not None else config.GALAXY10_H5_PATH
    if dest.exists() and not force:
        print(f"[data] Using existing file: {dest}")
        return dest

    print(f"[data] Downloading Galaxy10 DECaLS from:\n  {config.GALAXY10_URL}")
    print(f"[data] Saving to: {dest}")
    # Large file (~2.5GB+); show progress via urlretrieve reporthook.
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * downloaded / total_size)
            print(f"\r[data] {pct:5.1f}% ({downloaded / 1e6:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(config.GALAXY10_URL, dest, reporthook=_progress)
    print("\n[data] Download complete.")
    return dest


def load_galaxy10_arrays(h5_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels from Galaxy10_DECals.h5.

    Returns
    -------
    images : np.ndarray
        uint8 array of shape (N, H, W, 3)
    labels : np.ndarray
        int labels of shape (N,)
    """
    path = Path(h5_path) if h5_path is not None else config.GALAXY10_H5_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Galaxy10 file not found at {path}. Run download_galaxy10() first."
        )

    with h5py.File(path, "r") as f:
        # Galaxy10 DECaLS schema: 'images', 'ans' (labels)
        images = np.array(f["images"])
        labels = np.array(f["ans"]).astype(np.int64)

    print(f"[data] Loaded images={images.shape}, labels={labels.shape}")
    return images, labels


def stratified_split(
    labels: np.ndarray,
    seed: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train / validation / test index splits.

    Ratios follow config.TRAIN_RATIO / VAL_RATIO / TEST_RATIO.
    """
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - config.TRAIN_RATIO),
        random_state=seed,
        stratify=labels,
    )
    # Split remaining into val/test with relative proportions.
    relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        random_state=seed,
        stratify=labels[temp_idx],
    )
    return train_idx, val_idx, test_idx


class Galaxy10Dataset(Dataset):
    """
    Torch dataset wrapping Galaxy10 image/label arrays.

    Images are expected as (N, H, W, C) uint8; returned as transformed tensors.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform=None,
    ):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        image = self.images[idx]  # HWC uint8
        label = int(self.labels[idx])

        # Convert to PIL-compatible HWC array for torchvision transforms.
        from PIL import Image

        pil_image = Image.fromarray(image.astype(np.uint8))
        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)
        return image_tensor, label


def get_vae_transforms(image_size: int = config.IMAGE_SIZE_VAE):
    """VAE inputs in [0, 1] (sigmoid decoder), resized for tractable training."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
        ]
    )


def get_cnn_transforms(
    image_size: int = config.IMAGE_SIZE_CNN,
    train: bool = True,
    imagenet_norm: bool = True,
):
    """
    CNN preprocessing.

    ResNet50 (default): ImageNet normalisation (Lab 2).
    SimpleCNN smoke path: resize + ToTensor only (imagenet_norm=False).
    """
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if imagenet_norm:
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(ops)


def make_dataloaders(
    images: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mode: str = "cnn",
    batch_size: Optional[int] = None,
    cnn_image_size: Optional[int] = None,
    cnn_imagenet_norm: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders for either 'cnn' or 'vae' preprocessing.
    """
    if mode == "cnn":
        bs = batch_size or config.CNN_BATCH_SIZE
        size = cnn_image_size or config.IMAGE_SIZE_CNN
        train_ds = Galaxy10Dataset(
            images,
            labels,
            train_idx,
            get_cnn_transforms(image_size=size, train=True, imagenet_norm=cnn_imagenet_norm),
        )
        val_ds = Galaxy10Dataset(
            images,
            labels,
            val_idx,
            get_cnn_transforms(image_size=size, train=False, imagenet_norm=cnn_imagenet_norm),
        )
        test_ds = Galaxy10Dataset(
            images,
            labels,
            test_idx,
            get_cnn_transforms(image_size=size, train=False, imagenet_norm=cnn_imagenet_norm),
        )
    elif mode == "vae":
        bs = batch_size or config.VAE_BATCH_SIZE
        tfm = get_vae_transforms()
        train_ds = Galaxy10Dataset(images, labels, train_idx, tfm)
        val_ds = Galaxy10Dataset(images, labels, val_idx, tfm)
        test_ds = Galaxy10Dataset(images, labels, test_idx, tfm)
    else:
        raise ValueError("mode must be 'cnn' or 'vae'")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def class_counts(labels: np.ndarray, indices: np.ndarray) -> dict:
    """Return {class_id: count} for the selected indices."""
    subset = labels[indices]
    return {int(c): int(np.sum(subset == c)) for c in range(config.NUM_CLASSES)}
