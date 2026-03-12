# Transformer Anomaly Detection

Unsupervised time series anomaly detection with transformer autoencoders.

## Models

- **Lightweight** - shared encoder
- **Full** - encoder–decoder

Both available in different sizes.

## Feature extraction

You can use trained autoencoders as vectorizers for downstream models.

- `pipeline.extract_features(...)` returns one vector per window.
- Supported pooling: `mean`, `max`, `last`, `flatten`.
- `inference.extract_vectors_from_file(...)` can directly load model/data and optionally save vectors (`.npy` or `.npz`).
