"""
End-to-end CRISP-DM pipeline for the Galaxy10 CNN + VAE study.

Phases covered:
  1) Business Understanding  -> research question (docstring / prints)
  2) Data Understanding      -> load Galaxy10, class counts
  3) Data Preparation        -> stratified splits, transforms
  4) Modeling                -> train VAE; train CNN (real-only and real+synth)
  5) Evaluation              -> metrics, confusion matrices, minority focus
  6) Deployment              -> saved models/metrics/figures for the paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as `python -m src.run_pipeline` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data import (
    class_counts,
    download_galaxy10,
    ensure_dirs,
    get_cnn_transforms,
    load_galaxy10_arrays,
    make_dataloaders,
    stratified_split,
)
from src.evaluate import compare_minority_performance, evaluate_classifier
from src.generate import build_balanced_synthetic_arrays, make_mixed_loader
from src.train_cnn import train_cnn
from src.train_vae import load_vae, train_vae
from src.visualize import (
    plot_class_counts,
    plot_per_class_f1_comparison,
    plot_reconstructions,
)


def parse_args():
    p = argparse.ArgumentParser(description="Galaxy10 DECaLS CNN+VAE CRISP-DM pipeline")
    p.add_argument("--skip-download", action="store_true", help="Do not attempt dataset download")
    p.add_argument("--vae-epochs", type=int, default=config.VAE_EPOCHS)
    p.add_argument("--cnn-head-epochs", type=int, default=config.CNN_HEAD_EPOCHS)
    p.add_argument("--cnn-ft-epochs", type=int, default=config.CNN_FINETUNE_EPOCHS)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run (small real subset, or synthetic data if H5 missing)",
    )
    p.add_argument(
        "--resume-vae",
        action="store_true",
        help="Skip VAE training and load outputs/models/vae_galaxy10.pt",
    )
    return p.parse_args()


def _make_synthetic_smoke_data(n_per_class: int = 40):
    """Create a tiny fake RGB dataset so the pipeline can be smoke-tested offline."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    images, labels = [], []
    for c in range(config.NUM_CLASSES):
        # Deterministic colour bias per class so the CNN has a learnable signal.
        base = np.zeros((n_per_class, 64, 64, 3), dtype=np.uint8)
        base[..., c % 3] = 80 + 15 * c
        noise = rng.integers(0, 40, size=base.shape, dtype=np.uint8)
        images.append(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        labels.append(np.full(n_per_class, c, dtype=np.int64))
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def main():
    args = parse_args()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pipeline] device={device}")

    # ------------------------------------------------------------------
    # CRISP-DM: Business Understanding
    # ------------------------------------------------------------------
    print(
        "\n[Business Understanding]\n"
        "RQ: Given class imbalance in Galaxy10 DECaLS, do VAE-generated\n"
        "synthetic images improve CNN morphological classification on\n"
        "minority classes without degrading majority-class performance?\n"
    )

    # ------------------------------------------------------------------
    # CRISP-DM: Data Understanding / Data Preparation
    # ------------------------------------------------------------------
    if args.smoke and not config.GALAXY10_H5_PATH.exists():
        print("[smoke] Galaxy10 H5 not found — using synthetic offline smoke data.")
        images, labels = _make_synthetic_smoke_data()
    else:
        if not args.skip_download:
            download_galaxy10()
        images, labels = load_galaxy10_arrays()

        if args.smoke:
            # Keep a stratified ~1200-image subset for a quick end-to-end check.
            rng = np.random.default_rng(config.RANDOM_SEED)
            keep = []
            for c in range(config.NUM_CLASSES):
                idx = np.where(labels == c)[0]
                take = min(len(idx), 120)
                keep.append(rng.choice(idx, size=take, replace=False))
            keep = np.concatenate(keep)
            images, labels = images[keep], labels[keep]
            print(f"[smoke] subset size={len(labels)}")

    train_idx, val_idx, test_idx = stratified_split(labels)
    counts = class_counts(labels, train_idx)
    print("[Data Understanding] train class counts:", counts)
    print("[Data Understanding] minority focus:", config.MINORITY_CLASS_IDS)
    plot_class_counts(counts, title="Train split class counts (Galaxy10 DECaLS)")

    # Smoke runs use fewer epochs by default when the user did not override.
    vae_epochs = args.vae_epochs if not args.smoke else min(args.vae_epochs, 2)
    head_epochs = args.cnn_head_epochs if not args.smoke else min(args.cnn_head_epochs, 1)
    ft_epochs = args.cnn_ft_epochs if not args.smoke else min(args.cnn_ft_epochs, 1)
    if args.smoke:
        # Avoid accidental long defaults when flags were left at config values.
        if args.vae_epochs == config.VAE_EPOCHS:
            vae_epochs = 2
        if args.cnn_head_epochs == config.CNN_HEAD_EPOCHS:
            head_epochs = 1
        if args.cnn_ft_epochs == config.CNN_FINETUNE_EPOCHS:
            ft_epochs = 1

    vae_train, vae_val, _ = make_dataloaders(
        images, labels, train_idx, val_idx, test_idx, mode="vae"
    )

    # Full experiment: ResNet50 @ 224 + ImageNet norm (Lab 2).
    # Smoke: SimpleCNN @ 64 without ImageNet weights (fast CPU path).
    backbone = "simple" if args.smoke else "resnet50"
    cnn_size = config.IMAGE_SIZE_VAE if args.smoke else config.IMAGE_SIZE_CNN
    cnn_imagenet = not args.smoke
    cnn_train, cnn_val, cnn_test = make_dataloaders(
        images,
        labels,
        train_idx,
        val_idx,
        test_idx,
        mode="cnn",
        cnn_image_size=cnn_size,
        cnn_imagenet_norm=cnn_imagenet,
    )
    print(f"[Modeling] CNN backbone={backbone} image_size={cnn_size}")

    # ------------------------------------------------------------------
    # CRISP-DM: Modeling — VAE (Generative AI component)
    # ------------------------------------------------------------------
    vae_path = config.OUTPUT_MODELS / "vae_galaxy10.pt"
    if args.resume_vae:
        if not vae_path.exists():
            raise FileNotFoundError(f"--resume-vae set but checkpoint missing: {vae_path}")
        print(f"\n[Modeling] Loading ConvVAE from {vae_path}")
        vae = load_vae(vae_path, device=device)
    else:
        print("\n[Modeling] Training ConvVAE (generative model)...")
        vae = train_vae(
            vae_train,
            vae_val,
            epochs=vae_epochs,
            device=device,
            save_path=vae_path,
        )
    # Save reconstruction grid for the paper.
    sample_batch, _ = next(iter(vae_val))
    plot_reconstructions(vae, sample_batch)

    print("\n[Modeling] Building real+synthetic training set via VAE sampling...")
    syn_images, syn_labels, added = build_balanced_synthetic_arrays(
        images, labels, train_idx, vae, device
    )
    print("[Modeling] synthetics added per class:", added)

    cnn_train_aug = make_mixed_loader(
        images=images,
        labels=labels,
        train_idx=train_idx,
        syn_images=syn_images,
        syn_labels=syn_labels,
        transform=get_cnn_transforms(
            image_size=cnn_size, train=True, imagenet_norm=cnn_imagenet
        ),
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=True,
    )

    # ------------------------------------------------------------------
    # CRISP-DM: Modeling — CNN baselines / treatments
    # ------------------------------------------------------------------
    use_pretrained = (not args.smoke) and (backbone == "resnet50")

    print("\n[Modeling] Training CNN on REAL-ONLY data...")
    cnn_real = train_cnn(
        cnn_train,
        cnn_val,
        save_path=config.OUTPUT_MODELS / "cnn_real_only.pt",
        head_epochs=head_epochs,
        finetune_epochs=ft_epochs,
        device=device,
        pretrained=use_pretrained,
        backbone=backbone,
    )

    print("\n[Modeling] Training CNN on REAL + VAE SYNTHETIC data...")
    cnn_aug = train_cnn(
        cnn_train_aug,
        cnn_val,
        save_path=config.OUTPUT_MODELS / "cnn_real_plus_vae.pt",
        head_epochs=head_epochs,
        finetune_epochs=ft_epochs,
        device=device,
        pretrained=use_pretrained,
        backbone=backbone,
    )

    # ------------------------------------------------------------------
    # CRISP-DM: Evaluation
    # ------------------------------------------------------------------
    print("\n[Evaluation] Held-out test set comparison...")
    metrics_real = evaluate_classifier(cnn_real, cnn_test, device, tag="real_only")
    metrics_aug = evaluate_classifier(cnn_aug, cnn_test, device, tag="real_plus_vae")
    compare_minority_performance(metrics_real, metrics_aug)
    plot_per_class_f1_comparison(metrics_real, metrics_aug)

    print("\n[Deployment] Artifacts saved under outputs/models, outputs/metrics, outputs/figures")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
