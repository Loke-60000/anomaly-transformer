from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.model import create_model, BaseAutoencoder
from .data.data_loader import AnomalyDataLoader
from .training.train import (
    AnomalyDetectionTrainer,
    TrainingConfig,
    TrainingHistory,
)

MODEL_PRESETS: Dict[str, Dict] = {
    "pico": {
        "lightweight": dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=192, bottleneck_ratio=4),
        "full": dict(
            d_model=96,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
        ),
    },
    "medium": {
        "lightweight": dict(d_model=96, nhead=4, num_layers=3, dim_feedforward=384, bottleneck_ratio=4),
        "full": dict(
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
        ),
    },
    "ookii": {
        "lightweight": dict(d_model=128, nhead=4, num_layers=4, dim_feedforward=512, bottleneck_ratio=4),
        "full": dict(
            d_model=192,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=768,
        ),
    },
}

def build_model(
    model_type: str = "lightweight",
    input_dim: int = 1,
    preset: str = "medium",
    **overrides,
) -> BaseAutoencoder:
    if preset not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from {list(MODEL_PRESETS)}"
        )

    kwargs: dict = dict(input_dim=input_dim, dropout=0.1)
    kwargs.update(MODEL_PRESETS[preset][model_type])
    kwargs.update(overrides)

    if model_type == "full" and "activation" not in kwargs:
        kwargs["activation"] = "gelu"

    return create_model(model_type, **kwargs)


def load_checkpoint(
    checkpoint_path: str | Path,
    model_type: str = "lightweight",
    input_dim: int = 1,
    preset: str = "medium",
    device: Optional[str] = None,
    **model_overrides,
) -> BaseAutoencoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_type, input_dim=input_dim, preset=preset, **model_overrides
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    model.eval()

    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", None)
        vl = f"{val_loss:.6f}" if val_loss is not None else "n/a"
        print(f"Loaded checkpoint (epoch {epoch}, val_loss {vl})")

    return model

def train(
    model: BaseAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    checkpoint_dir: str = "./checkpoints",
    device: Optional[str] = None,
    live_plot: bool = True,
) -> Tuple[AnomalyDetectionTrainer, TrainingHistory]:
    config = TrainingConfig(
        learning_rate=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        early_stopping_patience=patience,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = AnomalyDetectionTrainer(model=model, config=config, device=device)
    history = trainer.fit(train_loader, val_loader, live_plot=live_plot)
    return trainer, history

def evaluate(
    model: BaseAutoencoder,
    test_loader: DataLoader,
    device: Optional[str] = None,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    scores: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                data = batch[0]
                batch_labels = batch[1] if len(batch) > 1 else None
            else:
                data = batch
                batch_labels = None

            data = data.to(device)
            recon = model(data)
            err = torch.mean(torch.abs(recon - data), dim=(1, 2))
            scores.extend(err.cpu().numpy().tolist())

            if batch_labels is not None:
                labels.extend(batch_labels.cpu().numpy().tolist())

    result: Dict = {"scores": np.array(scores)}

    if labels:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

        y_true, y_scores = np.array(labels), np.array(scores)
        try:
            result["auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            result["auc"] = None

        # Find the threshold that maximizes F1 instead of using a fixed
        # percentile — much better when ground-truth labels are available.
        best_f1, best_thr = 0.0, float(np.percentile(y_scores, 95))
        for pct in range(80, 100):
            thr_candidate = float(np.percentile(y_scores, pct))
            preds = (y_scores > thr_candidate).astype(int)
            _, _, f1_candidate, _ = precision_recall_fscore_support(
                y_true, preds, average="binary", zero_division=0
            )
            if f1_candidate >= best_f1:
                best_f1, best_thr = f1_candidate, thr_candidate

        threshold = best_thr
        y_pred = (y_scores > threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        result.update(
            threshold=threshold,
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
        )

    return result

def detect(
    model: BaseAutoencoder,
    data: np.ndarray,
    *,
    data_loader: Optional[AnomalyDataLoader] = None,
    window_size: int = 50,
    stride: int = 1,
    threshold_percentile: float = 95.0,
    threshold_method: str = "gaussian",
    threshold_kwargs: Optional[Dict] = None,
    train_split: float = 0.8,
    device: Optional[str] = None,
) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if data.ndim == 3:
        model = model.to(device)
        model.eval()
        all_scores: list[float] = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(data), bs):
                batch = torch.FloatTensor(data[i : i + bs]).to(device)
                recon = model(batch)
                err = torch.mean(torch.abs(batch - recon), dim=(1, 2))
                all_scores.extend(err.cpu().numpy().tolist())

        scores = np.array(all_scores)

        from .inference.inference import create_threshold_strategy

        if threshold_kwargs is None:
            threshold_kwargs = {}
        if threshold_method == "percentile" and "percentile" not in threshold_kwargs:
            threshold_kwargs["percentile"] = threshold_percentile
        strategy = create_threshold_strategy(threshold_method, **threshold_kwargs)
        strategy.fit(scores)
        thr = strategy.threshold

        anomalies = scores > thr
        return dict(
            scores=scores,
            anomalies=anomalies,
            threshold=thr,
            threshold_info=strategy.to_dict(),
            n_anomalies=int(anomalies.sum()),
            anomaly_rate=float(anomalies.mean()),
        )

    if data_loader is None:
        raise ValueError(
            "data_loader is required for raw time-series detection "
            "(it holds the fitted scaler)."
        )

    from .inference.inference import AnomalyDetector

    detector = AnomalyDetector(
        model=model,
        data_loader=data_loader,
        threshold_percentile=threshold_percentile,
        threshold_method=threshold_method,
        threshold_kwargs=threshold_kwargs,
        device=device,
    )

    split_idx = int(len(data_loader.processed_data) * train_split)
    detector.fit_threshold(data_loader.processed_data[:split_idx], window_size, stride)

    return detector.detect_anomalies(data, window_size, stride)


def extract_features(
    model: BaseAutoencoder,
    data: np.ndarray,
    *,
    data_loader: Optional[AnomalyDataLoader] = None,
    window_size: int = 50,
    stride: int = 1,
    batch_size: int = 64,
    pooling: str = "mean",
    device: Optional[str] = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if data.ndim == 3:
        from .inference.inference import AnomalyDetector

        detector = AnomalyDetector(
            model=model,
            data_loader=data_loader,
            threshold_percentile=95.0,
            device=device,
        )
        return detector.extract_feature_vectors(
            data=data,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            pooling=pooling,
        )

    if data_loader is None:
        raise ValueError(
            "data_loader is required for raw time-series feature extraction "
            "(it holds the fitted scaler)."
        )

    from .inference.inference import AnomalyDetector

    detector = AnomalyDetector(
        model=model,
        data_loader=data_loader,
        threshold_percentile=95.0,
        device=device,
    )

    normalized = data_loader.scaler.transform(data.reshape(-1, 1)).flatten()
    return detector.extract_feature_vectors(
        data=normalized,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        pooling=pooling,
    )

def load_timeseries(
    json_path: str,
    *,
    window_size: int = 50,
    stride: int = 1,
    batch_size: int = 32,
    train_split: float = 0.8,
    data_source: str = "nodes",
    unit_id: str = "73",
) -> Tuple[DataLoader, DataLoader, np.ndarray, AnomalyDataLoader]:
    dl = AnomalyDataLoader(
        json_path=json_path,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        train_split=train_split,
        normalize=True,
        data_source=data_source,
        unit_id=unit_id,
    )
    train_loader, val_loader, raw_data = dl.load_and_process()
    return train_loader, val_loader, raw_data, dl
