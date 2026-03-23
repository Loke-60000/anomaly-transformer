# Transformer Anomaly Detection

Unsupervised time series anomaly detection with transformer autoencoders.

## Models

- **Lightweight** — shared encoder + tight bottleneck (`d_model/4` compression with LayerNorm)
- **Full** — encoder–decoder with **learned decoder queries**

Both available in three size presets: `pico`, `medium`, `ookii`.

## Thresholding

Three strategies for setting the anomaly threshold, selected via `threshold_method`:

| Method       | Description                                       | Default                         |
| ------------ | ------------------------------------------------- | ------------------------------- |
| `gaussian`   | `mean + k × sigma` on training errors             | **default**, `k=3`              |
| `pot`        | Peaks Over Threshold (Extreme Value Theory / GPD) | strictest, best for heavy tails |
| `percentile` | Fixed percentile of training errors               | legacy, `p=95`                  |

## Feature extraction

Use trained autoencoders as vectorizers for downstream models.

- `pipeline.extract_features(...)` returns one vector per window.
- Supported pooling: `mean`, `max`, `last`, `flatten`.
- `inference.extract_vectors_from_file(...)` can directly load model/data and optionally save vectors (`.npy` or `.npz`).
