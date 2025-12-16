#!/usr/bin/env python3
"""Train the *baseline* (non-improved) transformer on NASA SMAP/MSL processed data.

This uses the models in `src/models/model.py` (create_model: lightweight/full)
and the generic trainer in `src/training/train.py`.

Example:
    python src/scripts/train_baseline_nasa.py --model-type lightweight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.model import create_model
from src.training.train import AnomalyDetectionTrainer, TrainingConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline transformer (non-improved) on NASA processed .npz"
    )
    parser.add_argument(
        "--data",
        default=str(
            PROJECT_ROOT / "assets" / "data" / "nasa" / "nasa_processed_data.npz"
        ),
        help="Path to nasa_processed_data.npz",
    )
    parser.add_argument(
        "--model-type",
        choices=["lightweight", "full"],
        default="lightweight",
        help="Baseline architecture variant",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "checkpoints_baseline_nasa"),
        help="Where to save baseline checkpoints",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cpu' or 'cuda'. Default: auto",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"NASA data not found: {data_path}")

    npz = np.load(data_path)
    train_sequences = npz["train_sequences"]  # [N, T, F]

    if train_sequences.ndim != 3:
        raise ValueError(
            f"Expected train_sequences to have shape [N, T, F], got {train_sequences.shape}"
        )

    n_samples, seq_len, n_features = train_sequences.shape

    # Simple train/val split
    split_idx = int(n_samples * float(args.train_split))
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError(
            f"Invalid --train-split={args.train_split}. Must keep some samples for both train/val."
        )

    train_arr = torch.tensor(train_sequences[:split_idx], dtype=torch.float32)
    val_arr = torch.tensor(train_sequences[split_idx:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_arr),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_arr),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Baseline model configs (kept modest; you can tune)
    if args.model_type == "lightweight":
        model = create_model(
            "lightweight",
            input_dim=n_features,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )
    else:
        model = create_model(
            "full",
            input_dim=n_features,
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
        )

    config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        early_stopping_patience=15,
        grad_clip_norm=1.0,
        checkpoint_dir=args.checkpoint_dir,
    )

    device = args.device
    trainer = AnomalyDetectionTrainer(model=model, config=config, device=device)

    print("=" * 70)
    print("BASELINE TRANSFORMER TRAINING (NASA)")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Train sequences: {split_idx} | Val sequences: {n_samples - split_idx}")
    print(f"Sequence length: {seq_len} | Features: {n_features}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    history = trainer.fit(train_loader, val_loader)

    print("\n✓ Done")
    print(f"Best val loss: {trainer.best_val_loss:.6f}")
    print(f"Best checkpoint: {Path(args.checkpoint_dir) / 'best_model.pt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
