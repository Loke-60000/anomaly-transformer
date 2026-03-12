from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def download_nasa(
    output_path: str | Path = "assets/data/nasa/nasa_processed_data.npz",
    window_size: int = 50,
    stride: int = 1,
    max_channels: Optional[int] = None,
) -> Path:
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required for downloading.  Install it with:\n"
            "  pip install kagglehub"
        )

    output_path = Path(output_path)
    if output_path.exists():
        print(f"Already exists: {output_path}  (delete to re-download)")
        return output_path.resolve()

    print("Downloading NASA SMAP/MSL dataset from Kaggle …")
    dataset_dir = Path(
        kagglehub.dataset_download(
            "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
        )
    )
    print(f"Raw data at: {dataset_dir}")

    import csv

    label_csv = dataset_dir / "labeled_anomalies.csv"
    if not label_csv.exists():
        # some kagglehub versions nest the data one level deeper
        candidates = list(dataset_dir.rglob("labeled_anomalies.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"labeled_anomalies.csv not found under {dataset_dir}"
            )
        label_csv = candidates[0]

    with open(label_csv, "r") as f:
        reader = csv.DictReader(f)
        channels = list(reader)

    if max_channels:
        channels = channels[:max_channels]

    print(
        f"Processing {len(channels)} channels (window={window_size}, stride={stride}) …"
    )

    train_dir = test_dir = None
    for candidate in dataset_dir.rglob("train"):
        if candidate.is_dir():
            train_dir = candidate
            break
    for candidate in dataset_dir.rglob("test"):
        if candidate.is_dir():
            test_dir = candidate
            break

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not find train/ and test/ directories under {dataset_dir}"
        )

    all_train_windows: List[np.ndarray] = []
    all_test_windows: List[np.ndarray] = []
    all_test_labels: List[np.ndarray] = []
    channel_info: list = []

    for ch in channels:
        chan_id = ch["chan_id"]
        train_npy = train_dir / f"{chan_id}.npy"
        test_npy = test_dir / f"{chan_id}.npy"

        if not train_npy.exists() or not test_npy.exists():
            print(f"  skip {chan_id} (files missing)")
            continue

        train_data = np.load(train_npy).astype(np.float32)
        test_data = np.load(test_npy).astype(np.float32)

        anomaly_seqs = eval(ch["anomaly_sequences"])  # list of [start, end]
        test_labels_raw = np.zeros(len(test_data), dtype=np.float32)
        for start, end in anomaly_seqs:
            test_labels_raw[start : min(end, len(test_data))] = 1.0

        tw = _window(train_data, window_size, stride)
        tew = _window(test_data, window_size, stride)
        tlw = _window_labels(test_labels_raw, window_size, stride)

        if len(tw) > 0:
            all_train_windows.append(tw)
        if len(tew) > 0:
            all_test_windows.append(tew)
            all_test_labels.append(tlw)

        channel_info.append(
            dict(
                chan_id=chan_id,
                spacecraft=ch.get("spacecraft", ""),
                train_len=len(train_data),
                test_len=len(test_data),
                anomaly_ratio=float(test_labels_raw.mean()),
            )
        )
        print(f"  {chan_id}: train {tw.shape}, test {tew.shape}")

    train_sequences = np.concatenate(all_train_windows)
    test_sequences = np.concatenate(all_test_windows)
    test_labels = np.concatenate(all_test_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        test_labels=test_labels,
    )

    import json

    info_path = output_path.with_suffix(".json").with_name(
        output_path.stem + "_info.json"
    )
    with open(info_path, "w") as f:
        json.dump(channel_info, f, indent=2)

    print(f"\nSaved {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  train: {train_sequences.shape}")
    print(f"  test:  {test_sequences.shape}  labels: {test_labels.shape}")
    return output_path.resolve()


def _window(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if len(data) < window_size:
        return np.empty((0, window_size, data.shape[-1]), dtype=data.dtype)
    if data.ndim == 1:
        data = data[:, None]
    starts = range(0, len(data) - window_size + 1, stride)
    return np.stack([data[i : i + window_size] for i in starts])


def _window_labels(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    starts = range(0, len(labels) - window_size + 1, stride)
    return np.array(
        [1 if labels[i : i + window_size].any() else 0 for i in starts],
        dtype=np.int64,
    )


def load_nasa_npz(
    data_path: str | Path,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
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
