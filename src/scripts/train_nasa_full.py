#!/usr/bin/env python3
"""
Full NASA SMAP/MSL training script (GPU-ready).
Uses the shared loader/trainer and defaults to larger models/epochs for full training.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import load_nasa_npz
from src.models.model import create_model
from src.training.train import AnomalyDetectionTrainer, TrainingConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full NASA training (GPU preferred) with shared trainer"
    )
    parser.add_argument(
        "--data",
        default=str(
            PROJECT_ROOT / "assets" / "data" / "nasa" / "nasa_processed_data.npz"
        ),
        help="Path to nasa_processed_data.npz",
    )
    parser.add_argument("--model-type", choices=["lightweight", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "checkpoints_baseline_nasa"),
        help="Where to save checkpoints",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cuda' or 'cpu'. Default: auto-detect",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run reconstruction-based evaluation on the test split if available",
    )
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--pico",
        action="store_true",
        help="Tiny preset (short run, small model)",
    )
    preset_group.add_argument(
        "--ookii",
        action="store_true",
        help="Large preset (ookii = big in Japanese) for full-capacity training",
    )
    return parser


def create_model_for_full(model_type: str, input_dim: int, preset: str):
    """Build a configuration for full training with preset size."""
    if preset == "pico":
        light_cfg = {"d_model": 64, "nhead": 3, "num_layers": 2, "dim_feedforward": 192}
        full_cfg = {
            "d_model": 96,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
        }
    elif preset == "ookii":
        light_cfg = {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 512,
        }
        full_cfg = {
            "d_model": 192,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 768,
        }
    else:  # medium/default
        light_cfg = {"d_model": 96, "nhead": 4, "num_layers": 3, "dim_feedforward": 384}
        full_cfg = {
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dim_feedforward": 512,
        }

    if model_type == "lightweight":
        return create_model(
            "lightweight",
            input_dim=input_dim,
            d_model=light_cfg["d_model"],
            nhead=light_cfg["nhead"],
            num_layers=light_cfg["num_layers"],
            dim_feedforward=light_cfg["dim_feedforward"],
            dropout=0.1,
        )

    return create_model(
        "full",
        input_dim=input_dim,
        d_model=full_cfg["d_model"],
        nhead=full_cfg["nhead"],
        num_encoder_layers=full_cfg["num_encoder_layers"],
        num_decoder_layers=full_cfg["num_decoder_layers"],
        dim_feedforward=full_cfg["dim_feedforward"],
        dropout=0.1,
        activation="gelu",
    )


def evaluate_reconstruction(
    trainer: AnomalyDetectionTrainer, test_loader: torch.utils.data.DataLoader
) -> Dict:
    """Compute reconstruction metrics and optional AUC if labels are present."""
    trainer.model.eval()
    device = trainer.device
    scores: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                data = batch[0]
                batch_labels = batch[1] if len(batch) > 1 else None
            else:
                data = batch
                batch_labels = None

            data = data.to(device)
            recon = trainer.model(data)
            err = torch.mean(torch.abs(recon - data), dim=(1, 2))
            scores.extend(err.cpu().numpy().tolist())

            if batch_labels is not None:
                labels.extend(batch_labels.cpu().numpy().tolist())

    result = {"scores": scores}
    if labels:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

        y_true = np.array(labels)
        y_scores = np.array(scores)
        try:
            result["auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            result["auc"] = None

        threshold = float(np.percentile(y_scores, 95))
        y_pred = (y_scores > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        result.update(
            {
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return result


def main() -> int:
    args = build_arg_parser().parse_args()

    preset = "medium"
    if args.pico:
        preset = "pico"
        args.epochs = 10
        args.batch_size = 16
    elif args.ookii:
        preset = "ookii"
        args.epochs = 150
        args.batch_size = 96

    data_path = Path(args.data)
    train_loader, val_loader, test_loader, meta = load_nasa_npz(
        data_path=data_path, batch_size=args.batch_size, train_split=args.train_split
    )

    model = create_model_for_full(
        args.model_type, input_dim=meta["n_features"], preset=preset
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        early_stopping_patience=15,
        grad_clip_norm=1.0,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = AnomalyDetectionTrainer(model=model, config=config, device=device)

    print("=" * 70)
    print("FULL NASA TRAINING (GPU-READY)")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Preset: {preset}")
    print(f"Train sequences: {meta['n_train']} | Val sequences: {meta['n_val']}")
    print(f"Sequence length: {meta['seq_len']} | Features: {meta['n_features']}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {trainer.device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    history = trainer.fit(train_loader, val_loader)

    print("\n✓ Done")
    print(f"Best val loss: {trainer.best_val_loss:.6f}")

    results = None
    if args.eval and test_loader is not None:
        print("\nRunning evaluation on test split...")
        results = evaluate_reconstruction(trainer, test_loader)
        if "auc" in results:
            print(
                f"AUC: {results['auc'] if results['auc'] is not None else 'n/a'} | "
                f"F1: {results.get('f1', 'n/a')} | Precision: {results.get('precision', 'n/a')} | Recall: {results.get('recall', 'n/a')}"
            )
        else:
            print(
                "Test split available but no labels to score; recorded reconstruction errors only."
            )

    summary = {
        "data": str(data_path),
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_split": args.train_split,
        "best_val_loss": trainer.best_val_loss,
        "timestamp": datetime.now().isoformat(),
        "meta": meta,
        "device": trainer.device,
        "preset": preset,
    }
    if results:
        summary["evaluation"] = results

    summary_dir = PROJECT_ROOT / "assets" / "outputs" / "results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "full_train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_dir / 'full_train_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
