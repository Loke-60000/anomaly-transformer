from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def download_power(
    output_path: str | Path = "assets/data/power/power_processed_data.npz",
    window_size: int = 50,
    stride: int = 1,
    test_fraction: float = 0.15,
    resample: Optional[str] = "1h",
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

    print("Downloading Electric Power Consumption dataset from Kaggle …")
    dataset_dir = Path(
        kagglehub.dataset_download("fedesoriano/electric-power-consumption")
    )
    print(f"Raw data at: {dataset_dir}")

    csv_candidates = list(dataset_dir.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file found under {dataset_dir}")
    csv_path = csv_candidates[0]
    print(f"Loading {csv_path.name} …")

    import pandas as pd

    df = pd.read_csv(csv_path)

    if "Date" in df.columns and "Time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce"
        )
        df = df.drop(columns=["Date", "Time"])
    elif "Datetime" in df.columns or "datetime" in df.columns:
        dt_col = "Datetime" if "Datetime" in df.columns else "datetime"
        df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
        if dt_col != "datetime":
            df = df.drop(columns=[dt_col])

    if "datetime" in df.columns:
        df = df.set_index("datetime").sort_index()

    available = [c for c in _FEATURE_COLS if c in df.columns]
    if available:
        df = df[available]
    else:
        df = df.select_dtypes(include=[np.number])

    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    if resample and isinstance(df.index, pd.DatetimeIndex):
        df = df.resample(resample).mean()

    df = df.dropna()
    values = df.values.astype(np.float32)
    if len(values) == 0:
        raise ValueError("No valid rows after cleaning — check the dataset.")

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    values = (values - mean) / std

    print(
        f"Cleaned: {len(values)} rows × {values.shape[1]} features "
        f"(window={window_size}, stride={stride})"
    )

    split_idx = int(len(values) * (1 - test_fraction))
    train_raw = values[:split_idx]
    test_raw = values[split_idx:]

    train_sequences = _window(train_raw, window_size, stride)
    test_sequences = _window(test_raw, window_size, stride)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        feature_names=np.array(list(df.columns)),
        mean=mean,
        std=std,
    )

    print(f"\nSaved {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  train: {train_sequences.shape}")
    print(f"  test:  {test_sequences.shape}")
    return output_path.resolve()


def _window(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Slice 2-D array [T, F] into [N, window_size, F] windows."""
    if len(data) < window_size:
        return np.empty((0, window_size, data.shape[-1]), dtype=data.dtype)
    if data.ndim == 1:
        data = data[:, None]
    starts = range(0, len(data) - window_size + 1, stride)
    return np.stack([data[i : i + window_size] for i in starts])


def load_power_npz(
    data_path: str | Path,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Power data not found: {path}")

    data = np.load(path, allow_pickle=True)
    train_sequences = data["train_sequences"]
    test_sequences = data.get("test_sequences")

    if train_sequences.ndim != 3:
        raise ValueError(
            f"Expected train_sequences shape [N, T, F], got {train_sequences.shape}"
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
    if test_sequences is not None and len(test_sequences) > 0:
        test_tensor = torch.tensor(test_sequences, dtype=torch.float32)
        test_loader = DataLoader(
            TensorDataset(test_tensor),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    meta = {
        "n_train": split_idx,
        "n_val": n_samples - split_idx,
        "seq_len": seq_len,
        "n_features": n_features,
        "has_test": test_loader is not None,
        "has_labels": False,
    }

    return train_loader, val_loader, test_loader, meta
