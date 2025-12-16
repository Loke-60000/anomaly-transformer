"""
Helpers for loading the processed NASA SMAP/MSL dataset stored as numpy .npz.
Produces PyTorch DataLoaders for train/val/test splits.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_nasa_npz(
    data_path: str | Path,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    """Load nasa_processed_data.npz and return DataLoaders plus metadata."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"NASA data not found: {path}")

    data = np.load(path)
    train_sequences = data["train_sequences"]
    test_sequences = data.get("test_sequences")
    test_labels = data.get("test_labels")

    if train_sequences.ndim != 3:
        raise ValueError(
            f"Expected train_sequences to have shape [N, T, F], got {train_sequences.shape}"
        )

    n_samples, seq_len, n_features = train_sequences.shape
    split_idx = int(n_samples * float(train_split))
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError(
            f"Invalid train_split={train_split}. Must leave samples for both train and val."
        )

    train_tensor = torch.tensor(train_sequences[:split_idx], dtype=torch.float32)
    val_tensor = torch.tensor(train_sequences[split_idx:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = None
    if test_sequences is not None:
        test_tensor = torch.tensor(test_sequences, dtype=torch.float32)
        if test_labels is not None:
            label_tensor = torch.tensor(test_labels, dtype=torch.long)
            test_dataset = TensorDataset(test_tensor, label_tensor)
        else:
            test_dataset = TensorDataset(test_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

    meta = {
        "n_train": split_idx,
        "n_val": n_samples - split_idx,
        "seq_len": seq_len,
        "n_features": n_features,
        "has_test": test_loader is not None,
    }

    return train_loader, val_loader, test_loader, meta
