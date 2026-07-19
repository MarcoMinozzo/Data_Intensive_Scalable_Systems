"""Evaluation metrics and plotting for the CNN comparison study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src import config


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return y_true, y_pred arrays for a loader."""
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_true.append(labels.numpy())
        y_pred.append(preds)
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tag: str,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Compute overall and per-class metrics; optionally save JSON + confusion matrix.
    """
    save_dir = Path(save_dir) if save_dir else config.OUTPUT_METRICS
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = config.OUTPUT_FIGURES
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = predict(model, loader, device)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )

    results = {
        "tag": tag,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {
            str(i): {
                "name": config.CLASS_NAMES[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(config.NUM_CLASSES)
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)],
            digits=4,
            zero_division=0,
        ),
    }

    out_json = save_dir / f"metrics_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if k != "classification_report"}, f, indent=2)

    # Confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {tag}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig_path = fig_dir / f"cm_{tag}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"[eval] {tag}: acc={results['accuracy']:.4f} macro_f1={results['macro_f1']:.4f}")
    print(results["classification_report"])
    print(f"[eval] Wrote {out_json} and {fig_path}")
    return results


def compare_minority_performance(real_only: Dict, with_synth: Dict) -> None:
    """Print a focused comparison on minority classes (research-question lens)."""
    print("\n=== Minority-class comparison (research question) ===")
    for cid in config.MINORITY_CLASS_IDS:
        a = real_only["per_class"][str(cid)]
        b = with_synth["per_class"][str(cid)]
        print(
            f"Class {cid} ({a['name']}):\n"
            f"  real-only  f1={a['f1']:.4f} recall={a['recall']:.4f}\n"
            f"  +synthetic f1={b['f1']:.4f} recall={b['recall']:.4f}"
        )
    print(
        f"Overall macro-F1: real-only={real_only['macro_f1']:.4f} | "
        f"+synthetic={with_synth['macro_f1']:.4f}"
    )
