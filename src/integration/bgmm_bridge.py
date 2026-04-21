from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from bgmm import getModel


@dataclass
class Pipeline4Result:
    bgmm_scores: np.ndarray
    predictions: np.ndarray
    actuals: np.ndarray
    gmm: object


def _predict_sequence(
    transformer,
    features: np.ndarray,
    seq_len: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    N = len(features)
    n_preds = N - seq_len

    if n_preds <= 0:
        raise ValueError(
            f"Need at least seq_len + 1 = {seq_len + 1} feature vectors, got {N}."
        )

    windows = np.stack([features[i : i + seq_len] for i in range(n_preds)])

    preds = []
    for start in range(0, n_preds, batch_size):
        batch = torch.tensor(windows[start : start + batch_size]).to(device)
        with torch.no_grad():
            out = transformer(batch)
        preds.append(out.cpu().numpy())

    return np.concatenate(preds, axis=0)


def run_pipeline_4(
    features_6d: np.ndarray,
    transformer_6d,
    seq_len: int = 48,
    device: torch.device | None = None,
    batch_size: int = 256,
    train_split: float = 0.8,
    bgmm_start_components: int = 20,
    bgmm_class_portion_threshold: float = 0.003,
    bgmm_weight_concentration_prior: float = 1e-5,
) -> Pipeline4Result:
    if device is None:
        device = torch.device("cpu")

    transformer_6d.to(device)
    transformer_6d.eval()

    features_6d   = np.asarray(features_6d, dtype=np.float32)
    n_train_feats = int(len(features_6d) * train_split)

    predictions = _predict_sequence(transformer_6d, features_6d, seq_len, device, batch_size)
    actuals     = features_6d[seq_len:]

    # offset by seq_len so no test-split window enters the transformer context during BGMM training
    n_train     = n_train_feats - seq_len
    train_preds = predictions[:n_train][np.newaxis]  # (1, n_train, 6)

    gmm = getModel(
        train_preds,
        start_n_components=bgmm_start_components,
        class_portion_threshold=bgmm_class_portion_threshold,
        weight_concentration_prior=bgmm_weight_concentration_prior,
    )

    bgmm_scores = gmm.anomalityScore(predictions[np.newaxis])

    return Pipeline4Result(
        bgmm_scores=bgmm_scores,
        predictions=predictions,
        actuals=actuals,
        gmm=gmm,
    )
