"""Training utilities for CNN classifiers (ResNet50 or SimpleCNN)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.models.cnn import (
    build_resnet50_classifier,
    build_simple_cnn,
    unfreeze_layer4_and_fc,
)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    running, n_batches = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            running += loss.item()
            n_batches += 1
    return running / max(n_batches, 1)


def train_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: Path,
    head_epochs: int = config.CNN_HEAD_EPOCHS,
    finetune_epochs: int = config.CNN_FINETUNE_EPOCHS,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    backbone: str = "resnet50",
) -> nn.Module:
    """
    Train a CNN classifier.

    backbone='resnet50' (default, Lab 2):
      1) Train classifier head with frozen backbone
      2) Fine-tune layer4 + FC

    backbone='simple' (smoke / CPU):
      Train the full small CNN end-to-end for head_epochs + finetune_epochs.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    if backbone == "simple":
        model = build_simple_cnn(num_classes=config.NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.CNN_LR_HEAD)
        total_epochs = max(1, head_epochs + finetune_epochs)
        for epoch in range(1, total_epochs + 1):
            train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = _run_epoch(model, val_loader, criterion, None, device, train=False)
            print(f"[CNN-simple] epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state": model.state_dict(), "backbone": backbone}, save_path)
    elif backbone == "resnet50":
        model = build_resnet50_classifier(
            num_classes=config.NUM_CLASSES, pretrained=pretrained
        ).to(device)

        # Stage 1: head only
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=config.CNN_LR_HEAD)
        for epoch in range(1, head_epochs + 1):
            train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = _run_epoch(model, val_loader, criterion, None, device, train=False)
            print(f"[CNN-head] epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state": model.state_dict(), "backbone": backbone}, save_path)

        # Stage 2: fine-tune layer4 + fc
        unfreeze_layer4_and_fc(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=config.CNN_LR_FINETUNE)

        for epoch in range(1, finetune_epochs + 1):
            train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss = _run_epoch(model, val_loader, criterion, None, device, train=False)
            print(f"[CNN-ft] epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state": model.state_dict(), "backbone": backbone}, save_path)
    else:
        raise ValueError("backbone must be 'resnet50' or 'simple'")

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"[CNN] Saved checkpoint: {save_path}")
    return model


def load_cnn(
    path: Path,
    device: Optional[torch.device] = None,
    backbone: Optional[str] = None,
) -> nn.Module:
    """Load a trained CNN checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    backbone = backbone or ckpt.get("backbone", "resnet50")
    if backbone == "simple":
        model = build_simple_cnn(num_classes=config.NUM_CLASSES).to(device)
    else:
        model = build_resnet50_classifier(num_classes=config.NUM_CLASSES, pretrained=False).to(
            device
        )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
