# Repository Architecture

Complete technical documentation for **transformeranomalygen** — an unsupervised time-series anomaly detection system built on transformer autoencoders.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Core Idea](#core-idea)
4. [Data Pipeline](#data-pipeline)
5. [Models](#models)
6. [Training](#training)
7. [Inference & Detection](#inference--detection)
8. [Feature Extraction](#feature-extraction)
9. [Configuration System](#configuration-system)
10. [Visualization](#visualization)
11. [Pipeline API](#pipeline-api)
12. [Notebooks](#notebooks)
13. [Datasets](#datasets)
14. [Dependencies](#dependencies)

---

## Overview

This repo provides a complete pipeline for detecting anomalies in time-series data using **transformer-based autoencoders**. The core principle is unsupervised: the model learns to reconstruct "normal" time-series windows, and at inference time, windows that produce high reconstruction error are flagged as anomalous.

Two model architectures are offered:

- **Lightweight** — a shared-encoder transformer with a linear bottleneck (faster, fewer parameters).
- **Full** — a standard encoder–decoder transformer (more expressive, higher capacity).

The repo ships with support for three data sources:

- **NASA SMAP/MSL** — spacecraft telemetry with labeled anomalies (benchmark).
- **Electric Power Consumption** — household power measurements (unsupervised).
- **Custom JSON time-series** — proprietary dezem sensor data in a specific JSON schema.

---

## Directory Structure

```
transformeranomalygen/
├── README.md                     # Brief project overview
├── ARCHITECTURE.md               # This document
├── requirements.txt              # Python dependencies
│
├── src/                          # All source code
│   ├── __init__.py               # Public API re-exports
│   ├── pipeline.py               # High-level orchestration functions
│   ├── config/
│   │   └── config.py             # Dataclass-based configuration system
│   ├── data/
│   │   ├── data_loader.py        # JSON data loading, preprocessing, windowing
│   │   ├── nasa_npz.py           # NASA SMAP/MSL download & NPZ loader
│   │   └── power.py              # Power consumption download & NPZ loader
│   ├── models/
│   │   └── model.py              # Transformer autoencoder architectures
│   ├── training/
│   │   └── train.py              # Trainer, checkpointing, learning curves
│   ├── inference/
│   │   └── inference.py          # Anomaly detector, feature extraction
│   └── utils/
│       └── visualize.py          # Matplotlib/seaborn plotting utilities
│
├── notebooks/                    # Jupyter notebooks (per-dataset)
│   ├── nasa/
│   │   ├── train_lightweight.ipynb
│   │   ├── train_full.ipynb
│   │   └── inference.ipynb
│   └── power/
│       ├── train_lightweight.ipynb
│       ├── train_full.ipynb
│       └── inference.ipynb
│
├── checkpoints/                  # Saved model weights
│   ├── nasa/lightweight/best_model.pt
│   └── power/lightweight/best_model.pt
│
└── assets/
    ├── data/                     # Raw & processed datasets
    │   ├── nasa/
    │   ├── power/
    │   └── timeseries-data/      # Custom dezem JSON time-series
    ├── models/configs/
    │   └── my_config.json        # Example pipeline configuration
    └── outputs/
        ├── plots/
        └── results/
```

---

## Core Idea

```
Raw Time Series
      │
      ▼
┌─────────────────┐
│  Preprocessing   │  Clean, interpolate missing values, normalize (StandardScaler)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Windowing      │  Slide a fixed-size window (e.g. 50 steps, stride 1)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transformer     │  Autoencoder learns to reconstruct normal windows
│  Autoencoder     │  Loss = MSE(input, reconstruction)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Anomaly Score   │  score = MAE(input, reconstruction) per window
│  Thresholding    │  threshold = percentile of training scores (e.g. 95th)
└────────┬────────┘
         │
         ▼
   Anomalies flagged
```

**Training** is fully unsupervised — only normal (or mostly-normal) data is needed. The model learns the distribution of typical patterns. At **inference**, windows the model cannot reconstruct well receive high error scores and are classified as anomalous.

---

## Data Pipeline

### `src/data/data_loader.py`

This is the core data module for custom JSON time-series. It contains:

#### `DataPreprocessor`

Static utility class for data cleaning:

- **`clean_data()`** — removes non-finite values (NaN, Inf) via nearest-neighbor interpolation; optionally clips extreme outliers beyond `N` standard deviations.
- **`validate_data()`** — checks data length, variance, missing values; returns a validation report with statistics.
- **`interpolate_missing_values()`** — replaces sentinel missing values (default `[0, -1]`) with linear interpolation between surrounding valid points.

#### `TimeSeriesDataset` (PyTorch Dataset)

Takes a 1-D numpy array and creates sliding windows:

- `window_size` — length of each window (default 50).
- `stride` — step between consecutive windows (default 1).
- Returns `torch.FloatTensor` of shape `(window_size, 1)`.

#### Data Loaders

Three JSON loading strategies:

- **`JSONDataLoader`** — loads from a flat JSON structure: `data.values[0]` → list of `[value]` pairs.
- **`TimeSeriesNodeDataLoader`** — loads from dezem node JSON files: navigates `data.oCurveData.oData[unit_id].mResult` with timestamp-sorted measurements.
  - `load_multiple_nodes()` — loads from several node files and concatenates or averages them.
  - `load_from_index()` — reads an `index.json` that lists node file paths, then loads all successful nodes.

#### `AnomalyDataLoader`

The main data loader class that orchestrates everything:

1. Loads raw data via the appropriate JSON loader (controlled by `data_source`: `"json"`, `"nodes"`, or `"index"`).
2. Interpolates missing values.
3. Normalizes with `StandardScaler` (fit on all data).
4. Splits into train/val by `train_split` ratio (chronological split, no shuffling of split boundary).
5. Creates `TimeSeriesDataset` instances and wraps them in PyTorch `DataLoader`s.
6. Stores the fitted `scaler` for inverse transforms at inference time.

Returns: `(train_loader, val_loader, raw_data)`.

### `src/data/nasa_npz.py`

**`download_nasa()`** — downloads the NASA SMAP/MSL anomaly detection dataset from Kaggle via `kagglehub`, then:

1. Parses `labeled_anomalies.csv` for channel IDs and anomaly intervals.
2. Loads per-channel `.npy` train/test arrays.
3. Creates binary test labels from anomaly sequence annotations.
4. Windows all channels and concatenates into a single NPZ file with keys: `train_sequences`, `test_sequences`, `test_labels`.
5. Saves a companion `_info.json` with per-channel metadata.

**`load_nasa_npz()`** — loads the pre-processed NPZ file and returns:

- `train_loader` — training windows (first `train_split` fraction).
- `val_loader` — validation windows (remainder).
- `test_loader` — test windows with optional anomaly labels (as `TensorDataset` with two tensors when labels exist).
- `meta` dict — `n_train`, `n_val`, `seq_len`, `n_features`, `has_test`.

### `src/data/power.py`

**`download_power()`** — downloads the Electric Power Consumption dataset from Kaggle, then:

1. Parses the CSV (handles both `Date`+`Time` and `Datetime` column formats).
2. Selects 7 feature columns (power, voltage, sub-metering).
3. Optionally resamples to a coarser frequency (default `"1h"`).
4. Z-score normalizes each feature.
5. Windows and saves train/test splits to NPZ with keys: `train_sequences`, `test_sequences`, `feature_names`, `mean`, `std`.

**`load_power_npz()`** — analogous to `load_nasa_npz()` but without anomaly labels (unsupervised).

---

## Models

### `src/models/model.py`

#### `PositionalEncoding`

Standard sinusoidal positional encoding (from "Attention Is All You Need"). Adds position-dependent signals to the projected input embeddings so the transformer can reason about temporal order.

#### `BaseAutoencoder` (abstract)

Common interface for both architectures:

- `forward(src)` — full reconstruction pass (input → output of same shape).
- `encode(src)` — encoder-only pass (returns latent representations).
- `get_reconstruction_error(src, reduction)` — convenience method; returns MAE per sample.
- `_init_weights()` — Xavier uniform initialization for all parameters with dim > 1.

#### `TransformerAutoencoder` (Full)

A standard encoder–decoder transformer autoencoder:

```
Input (batch, seq_len, input_dim)
  → Linear projection to d_model
  → Positional encoding
  → TransformerEncoder (N layers)
  → TransformerDecoder (N layers, attending to encoder memory)
  → Linear projection back to input_dim
Output (batch, seq_len, input_dim)
```

Key parameters:

- `d_model` — hidden dimension (default 64).
- `nhead` — number of attention heads (default 4).
- `num_encoder_layers` / `num_decoder_layers` — depth (default 3 each).
- `dim_feedforward` — FFN inner dimension (default 256).
- `activation` — `"gelu"` by default.

The decoder uses the same positional-encoded input as both the target and the query, with encoder output as memory. This forces the model to learn compressed representations in the encoder.

#### `LightweightTransformerAutoencoder` (Lightweight)

A more efficient architecture using only a **shared encoder** plus a linear bottleneck:

```
Input (batch, seq_len, input_dim)
  → Linear projection to d_model
  → Positional encoding
  → TransformerEncoder (N layers, shared for encode & decode)
  → Bottleneck: Linear(d_model → d_model/2) → GELU → Linear(d_model/2 → d_model)
  → Linear projection back to input_dim
Output (batch, seq_len, input_dim)
```

The bottleneck forces information compression, similar to a classical autoencoder's latent space. This model is significantly faster and uses fewer parameters than the full variant.

#### `create_model(model_type, **kwargs)`

Factory function. `model_type` is `"lightweight"` or `"full"`.

### Model Presets

Defined in `pipeline.py` under `MODEL_PRESETS`:

| Preset     | Lightweight                      | Full                                 |
| ---------- | -------------------------------- | ------------------------------------ |
| **pico**   | d=64, heads=4, layers=2, ff=192  | d=96, heads=4, enc=2, dec=2, ff=256  |
| **medium** | d=96, heads=4, layers=3, ff=384  | d=128, heads=8, enc=3, dec=3, ff=512 |
| **ookii**  | d=128, heads=4, layers=4, ff=512 | d=192, heads=8, enc=4, dec=4, ff=768 |

`build_model(model_type, input_dim, preset, **overrides)` constructs a model from these presets, with optional parameter overrides.

---

## Training

### `src/training/train.py`

#### `TrainingConfig`

Simple configuration container:

- `learning_rate` (default 1e-3)
- `weight_decay` (default 1e-5)
- `epochs` (default 100)
- `early_stopping_patience` (default 15)
- `grad_clip_norm` (default 1.0)
- `checkpoint_dir` — auto-created on init.

#### `CheckpointManager`

Saves/loads full training state: model weights, optimizer state, scheduler state, epoch, validation loss, and training history.

#### `TrainingHistory`

Tracks per-epoch metrics:

- `train_loss`, `val_loss`, `learning_rates`, `epoch_times`.
- `summary_table()` — pretty-printed ASCII table.
- `display_table()` — HTML table for Jupyter environments.
- `plot_learning_curves()` — matplotlib loss + LR plots.
- Serializable to/from JSON.

#### `AnomalyDetectionTrainer`

The main training loop:

1. **Optimizer**: AdamW with configurable LR and weight decay.
2. **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=5 epochs).
3. **Loss**: MSE between input and reconstruction.
4. **Gradient clipping**: max norm = 1.0.
5. **Early stopping**: stops if validation loss doesn't improve for `patience` consecutive epochs.
6. **Checkpointing**: saves `best_model.pt` on every validation improvement, periodic checkpoints every 10 epochs, and `final_model.pt` at the end.
7. **Live plotting**: in Jupyter environments, updates a loss plot in real-time during training (via `IPython.display`).
8. **Progress bar**: uses `tqdm` with per-batch postfix updates.

The `fit()` method runs the full training loop and returns a `TrainingHistory` object.

#### Convenience Functions

- `AnomalyDetectionTrainer.from_simple_config()` — one-call setup: loads data, creates model, trains, returns everything.
- `train_anomaly_detector()` — wrapper around the above.

---

## Inference & Detection

### `src/inference/inference.py`

#### `AnomalyDetector`

The main inference class. Initialized with a trained model and a data loader (which holds the fitted scaler).

**`calculate_reconstruction_errors(data, window_size, stride)`**

- Takes raw or pre-windowed data.
- If raw: creates sliding windows, adds feature dimension if needed.
- Runs batched inference (default batch_size=64).
- Returns `(errors, reconstructions)` where errors is MAE per window.

**`fit_threshold(train_data, window_size, stride)`**

- Computes reconstruction errors on training data.
- Sets threshold at the configured percentile (e.g. 95th) of training errors.

**`detect_anomalies(data, window_size, stride)`**

- Normalizes raw data using the stored scaler.
- Computes per-window reconstruction errors.
- Flags windows exceeding the threshold.
- Maps window-level anomalies back to **point-level** anomaly scores by averaging overlapping window scores.
- Identifies contiguous anomaly segments.
- Returns a dict with:
  - `point_scores`, `point_anomalies` — per-timestep results.
  - `window_errors`, `window_anomalies` — per-window results.
  - `threshold`, `n_anomalies`, `anomaly_rate`.
  - `anomaly_segments` — list of `{start, end, length}` dicts.

**`analyze_anomalies(results, data)`**

- Generates a human-readable text report of detection results, including top anomaly segments.

**`save_results(results, output_path)`** — serializes results to JSON.

#### Static / Class Methods

- `load_model(checkpoint_path, model_type)` — loads a checkpoint with hardcoded default model config.
- `from_file(data_path, checkpoint_path, ...)` — complete detection pipeline from file paths: loads model → loads data → fits threshold → detects → prints report.

#### Standalone Functions

- `load_trained_model()` — backwards-compatible alias.
- `detect_anomalies_from_file()` — backwards-compatible alias.
- `extract_vectors_from_file()` — complete feature extraction pipeline from file paths.

---

## Feature Extraction

The trained autoencoder can be used as a **feature extractor** (vectorizer) for downstream tasks. The encoder transforms each input window into a sequence of latent vectors, which are then pooled into a single fixed-size vector.

**`AnomalyDetector.extract_feature_vectors(data, window_size, stride, pooling)`**

Pooling strategies:
| Strategy | Output | Description |
|----------|--------|-------------|
| `mean` | `(N, d_model)` | Average across time steps |
| `max` | `(N, d_model)` | Max-pool across time steps |
| `last` | `(N, d_model)` | Take the last time step only |
| `flatten` | `(N, seq_len * d_model)` | Concatenate all time steps |

The pipeline-level `extract_features()` function and `extract_vectors_from_file()` provide higher-level interfaces.

---

## Configuration System

### `src/config/config.py`

A dataclass-based configuration system with five config groups:

| Config                | Purpose       | Key Fields                                                                                    |
| --------------------- | ------------- | --------------------------------------------------------------------------------------------- |
| `DataConfig`          | Data loading  | `json_path`, `data_source`, `window_size`, `stride`, `batch_size`, `train_split`, `normalize` |
| `ModelConfig`         | Architecture  | `model_type`, `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`                  |
| `TrainingConfig`      | Training loop | `learning_rate`, `epochs`, `early_stopping_patience`, `checkpoint_dir`, `use_amp`             |
| `InferenceConfig`     | Detection     | `checkpoint_path`, `threshold_percentile`, `output_dir`                                       |
| `VisualizationConfig` | Plots         | `output_dir`, `style`, `dpi`, `save_format`                                                   |

#### `PipelineConfig`

Combines all five configs. Can be constructed from:

- `PipelineConfig.default(json_path)` — sensible defaults.
- `PipelineConfig.from_dict(config_dict)` — from a dictionary (e.g. loaded from JSON).

#### `ConfigurationManager`

- `save(filepath)` / `load(filepath)` — JSON serialization.
- `validate()` — runs all sub-config validators.
- `print_summary()` — formatted config report.

#### `PresetConfigurations`

Factory methods for common scenarios:

- `quick_test()` — tiny model, 10 epochs (for debugging).
- `balanced()` — default settings.
- `high_accuracy()` — full model, 200 epochs, AMP, 99th percentile threshold.
- `fast_training()` — small lightweight model, 50 epochs.
- `production()` — full model, AMP, dedicated output dirs.

An example config JSON is stored at `assets/models/configs/my_config.json`.

---

## Visualization

### `src/utils/visualize.py`

`AnomalyVisualizer` provides four plot types:

1. **`plot_time_series_with_anomalies()`** — two-panel chart:
   - Top: original time series with red scatter/shading on anomalous regions.
   - Bottom: anomaly scores with threshold line and filled regions above threshold.

2. **`plot_anomaly_distribution()`** — two-panel chart:
   - Left: histogram of anomaly scores with threshold line.
   - Right: box plots comparing normal vs. anomaly score distributions.

3. **`plot_training_history()`** — two-panel chart:
   - Left: train & validation loss curves.
   - Right: learning rate schedule (log scale).

4. **`plot_reconstruction()`** — overlay of original vs. reconstructed signal with error shading.

5. **`create_summary_report()`** — generates all relevant plots and saves to an output directory.

All plots support optional `save_path` for saving to disk (300 DPI PNGs by default).

---

## Pipeline API

### `src/pipeline.py`

This is the main user-facing API that ties everything together. All key functions are re-exported from `src/__init__.py`.

#### `build_model(model_type, input_dim, preset, **overrides)`

Creates a model from a preset configuration. Returns a `BaseAutoencoder`.

#### `load_checkpoint(checkpoint_path, model_type, input_dim, preset, device)`

Loads a trained model from a `.pt` checkpoint file. Handles both raw state dicts and full checkpoint dicts (with epoch, val_loss metadata).

#### `train(model, train_loader, val_loader, **kwargs)`

Trains a model with configurable hyperparameters. Returns `(trainer, history)`.

#### `evaluate(model, test_loader, device)`

Runs evaluation on a test set:

- Computes per-window reconstruction errors.
- If labels are present: calculates AUC, precision, recall, F1 (at 95th percentile threshold).

#### `detect(model, data, **kwargs)`

Detects anomalies in data:

- **If `data` is 3-D** (pre-windowed `[N, T, F]`): runs direct batched inference, thresholds at the given percentile.
- **If `data` is 1-D** (raw time series): requires a `data_loader` with fitted scaler; uses `AnomalyDetector` for full point-level detection with segment identification.

#### `extract_features(model, data, **kwargs)`

Extracts feature vectors from data using the encoder. Supports both pre-windowed and raw input.

#### `load_timeseries(json_path, **kwargs)`

Loads custom JSON time-series data and returns `(train_loader, val_loader, raw_data, data_loader)`.

---

## Notebooks

Six Jupyter notebooks provide complete workflows, organized by dataset:

### NASA SMAP/MSL

| Notebook                                 | Purpose                                                                                               |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `notebooks/nasa/train_lightweight.ipynb` | Download NASA data → build lightweight model (medium preset) → train 50 epochs → display history      |
| `notebooks/nasa/train_full.ipynb`        | Same flow with the full encoder–decoder model → train 100 epochs                                      |
| `notebooks/nasa/inference.ipynb`         | Load checkpoint → evaluate on test set (AUC, F1) → detect anomalies → visualize scores & distribution |

### Power Consumption

| Notebook                                  | Purpose                                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------------------- |
| `notebooks/power/train_lightweight.ipynb` | Download power data → build lightweight model (pico preset) → train 50 epochs          |
| `notebooks/power/train_full.ipynb`        | Same flow with full model (medium preset) → train 100 epochs                           |
| `notebooks/power/inference.ipynb`         | Load checkpoint → evaluate (reconstruction error only, no labels) → detect → visualize |

Each notebook follows the same pattern:

1. **Setup** — import from `src`, download/load data.
2. **Build** — create model via `build_model()` with preset.
3. **Train** or **Load** — train from scratch or load checkpoint.
4. **Evaluate** — compute metrics.
5. **Visualize** — plot results.

---

## Datasets

### NASA SMAP/MSL

- **Source**: Kaggle (`patrickfleith/nasa-anomaly-detection-dataset-smap-msl`)
- **Type**: Spacecraft telemetry (multi-channel, multivariate)
- **Labels**: Yes — anomaly intervals per channel in `labeled_anomalies.csv`
- **Stored as**: `assets/data/nasa/nasa_processed_data.npz`
- **Shape**: `[N, window_size, n_features]` — features vary by channel (typically 25)

### Electric Power Consumption

- **Source**: Kaggle (`fedesoriano/electric-power-consumption`)
- **Type**: Household power measurements
- **Features**: 7 (Global active/reactive power, Voltage, Global intensity, 3 sub-meters)
- **Labels**: None (unsupervised)
- **Stored as**: `assets/data/power/power_processed_data.npz`
- **Preprocessing**: hourly resampling, z-score normalization

### Custom JSON Time-Series (dezem)

- **Location**: `assets/data/timeseries-data/`
- **Format**: Node JSON files with measurement data keyed by unit ID and timestamp
- **Index**: `index.json` lists available node files
- **Loaded via**: `AnomalyDataLoader` with `data_source="nodes"` or `data_source="index"`

---

## Dependencies

| Package                        | Purpose                           |
| ------------------------------ | --------------------------------- |
| `torch >= 2.0`                 | Neural network framework          |
| `numpy >= 1.24`                | Array operations                  |
| `scikit-learn >= 1.3`          | StandardScaler, metrics (AUC, F1) |
| `matplotlib >= 3.7`            | Plotting                          |
| `seaborn >= 0.12`              | Plot styling                      |
| `tqdm >= 4.65`                 | Progress bars                     |
| `pandas >= 2.0`                | CSV parsing (power dataset)       |
| `kagglehub >= 0.3`             | Dataset downloads                 |
| `scipy >= 1.11`                | Scientific computing utilities    |
| `fastapi >= 0.100`             | API serving (optional)            |
| `uvicorn >= 0.23`              | ASGI server (optional)            |
| `jupyter, notebook, ipykernel` | Notebook support                  |
| `pytest, pytest-cov`           | Testing                           |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Train on NASA data

```python
from src import download_nasa, load_nasa_npz, build_model, train

data_path = download_nasa()
train_loader, val_loader, test_loader, meta = load_nasa_npz(data_path)

model = build_model("lightweight", input_dim=meta["n_features"], preset="medium")
trainer, history = train(model, train_loader, val_loader, epochs=50)
```

### Detect anomalies

```python
from src import load_checkpoint, detect
import numpy as np

model = load_checkpoint("checkpoints/nasa/lightweight/best_model.pt",
                        input_dim=meta["n_features"], preset="medium")

test_data = np.concatenate([b[0].numpy() for b in test_loader], axis=0)
results = detect(model, test_data, threshold_percentile=95.0)

print(f"Anomalies: {results['n_anomalies']} ({results['anomaly_rate']*100:.1f}%)")
```

### Extract features for downstream use

```python
from src import extract_features

vectors = extract_features(model, test_data, pooling="mean")
# vectors.shape → (n_windows, d_model)
```
