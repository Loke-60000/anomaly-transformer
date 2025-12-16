# Transformer Anomaly Detection

Quickstart for training and evaluating transformer autoencoders on the NASA SMAP/MSL dataset.

## Models

- Defined in src/models/model.py:
  - `lightweight`: single encoder reused for decoding via a bottleneck projection; fewer heads/layers for fast CPU runs.
  - `full`: separate encoder/decoder stacks with larger feedforward blocks for higher capacity.
- Shared pieces: sinusoidal positional encoding, input/output projection layers, Xavier init, reconstruction error helper for anomaly scoring.
- Selection: scripts pass `--model-type` (GPU script defaults to full) plus preset flags: `--pico` (tiny smoke), `--ookii` (large/long run).

## Architecture & Training Behavior

- Inputs: windowed sequences shaped [batch, time, features] (NASA default: time=50, features=25), normalized during preprocessing.
- Stem: linear projection to `d_model`, add sinusoidal positional encoding, dropout, then transformer blocks.
- Lightweight path: encoder-only stack, then a bottleneck MLP (d_model → d_model/2 → d_model) before projecting back to inputs. Lowest parameter count, CPU-friendly.
- Full path: encoder stack produces memory; decoder stack attends over memory to reconstruct inputs. Higher capacity for GPU runs.
- Reconstruction loss: MSE across time and features between input and reconstruction.
- Optimizer & schedule: AdamW; ReduceLROnPlateau (factor 0.5, patience 5); gradient clipping (max-norm 1.0).
- Early stopping & checkpoints: patience 15; best checkpoint on improvement, periodic every 10 epochs, and final model; training history JSON saved beside checkpoints.

## How the Model Detects Anomalies

- Train: fit the autoencoder to minimize reconstruction MSE on normal-ish windows.
- Score: use `get_reconstruction_error` to compute per-window error (mean/sum across time and features). Larger error ⇒ more anomalous.
- Threshold: pick a percentile on val/test scores (e.g., 95th) or tune by precision/recall when labels exist. Scripts can emit raw scores for custom thresholds.

## Prerequisites

- Python 3.10+ and PyTorch (with CUDA if using GPU)
- Processed data file: assets/data/nasa/nasa_processed_data.npz (generate via script below)

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

```bash
python scripts/data_processing/prepare_nasa_data.py
```

## Training Options

- CPU quick/long runs (small models, defaults to CPU):
  - `python src/scripts/train_dsa_cpu.py --device cpu --epochs 5  --eval` # tiny sanity
  - `python src/scripts/train_dsa_cpu.py --device cpu --epochs 50 --eval` # longer CPU run
- Full/GPU-ready (presets; default model type is full):
  - `python src/scripts/train_nasa_full.py --epochs 100 --eval`
  - `python src/scripts/train_nasa_full.py --pico  --eval` # tiny quick check
  - `python src/scripts/train_nasa_full.py --ookii --eval` # big/full run

Outputs: checkpoints under checkpoints_baseline_nasa/ or assets/models/checkpoints/, summaries under assets/outputs/results/.

## Notes

- Ensure the NASA npz exists before training.
- Use `--device cuda` on supporting hardware; otherwise defaults are CPU-friendly.
- Presets: `--pico` = tiny, `--ookii` = large.
