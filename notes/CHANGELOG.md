# Changelog

## 2026-03-13 — Model & Threshold Improvements

### Breaking

- **Existing checkpoints are incompatible** — retrain required. The bottleneck structure and full-model decoder changed, so old `.pt` files won't load.

---

### 1. Full model — learned decoder queries

**File**: `src/models/model.py` — `TransformerAutoencoder`

**Before**: The decoder received the same projected input as `tgt`, allowing it to bypass the encoder via self-attention — no real information bottleneck.

```python
# OLD
output = self.decoder(src, memory)  # src = raw projected input
```

**After**: The decoder uses **learned query tokens** (`nn.Parameter`). Reconstruction must pass entirely through the encoder's compressed `memory`.

```python
# NEW
tgt = self.decoder_queries[:seq_len].unsqueeze(1).expand(-1, batch_size, -1)
tgt = self.decoder_pos(tgt)
output = self.decoder(tgt, memory)
```

New parameter: `max_seq_len` (default 512) — controls the size of the learned query bank.

---

### 2. Lightweight model — tighter bottleneck

**File**: `src/models/model.py` — `LightweightTransformerAutoencoder`

**Before**: `d_model → d_model/2 → d_model` — barely compressing (e.g. 96 → 48 → 96).

**After**: `d_model → d_model/4 → d_model` with **LayerNorm** for training stability (e.g. 96 → 24 → 96).

```python
# OLD
nn.Linear(d_model, d_model // 2),
nn.GELU(),
nn.Linear(d_model // 2, d_model),

# NEW
nn.Linear(d_model, d_model // ratio),  # ratio=4 by default
nn.GELU(),
nn.LayerNorm(d_model // ratio),
nn.Linear(d_model // ratio, d_model),
```

New parameter: `bottleneck_ratio` (default 4) — configurable compression factor.

---

### 3. Threshold strategies

**File**: `src/inference/inference.py`

**Before**: Fixed 95th percentile — blindly flags ~5% of data regardless of actual distribution.

**After**: Three pluggable strategies via `threshold_method` parameter:

| Method                     | Class                 | How it works                                                               |
| -------------------------- | --------------------- | -------------------------------------------------------------------------- |
| `"gaussian"` (new default) | `GaussianThreshold`   | `mean + k × sigma` (default `k=3`)                                         |
| `"pot"`                    | `POTThreshold`        | Extreme Value Theory / Generalized Pareto Distribution on tail exceedances |
| `"percentile"`             | `PercentileThreshold` | Original behavior (backward compatible)                                    |

Factory: `create_threshold_strategy(method, **kwargs)`

All strategies implement: `fit(errors)`, `.threshold`, `to_dict()`

---

### 4. Smarter evaluation with labels

**File**: `src/pipeline.py` — `evaluate()`

**Before**: Fixed 95th percentile for F1/precision/recall calculation.

**After**: When ground-truth labels exist (e.g. NASA), sweeps percentiles 80–99 and selects the threshold that **maximizes F1**.

---

### 5. Pipeline wiring

**File**: `src/pipeline.py` — `detect()`

New parameters:

- `threshold_method` (default `"gaussian"`) — which strategy to use
- `threshold_kwargs` — passed through to the strategy constructor

The result dict now includes `threshold_info` with the strategy's metadata.

---

### 6. Updated presets

**File**: `src/pipeline.py` — `MODEL_PRESETS`

All lightweight presets now include `bottleneck_ratio=4`.

---

### 7. Updated notebooks

**Files**: `notebooks/nasa/inference.ipynb`, `notebooks/power/inference.ipynb`

Detection cells updated to use `threshold_method="gaussian"` and display `threshold_info`.

---

### 8. Updated exports

**File**: `src/inference/__init__.py`

New public exports: `ThresholdStrategy`, `PercentileThreshold`, `GaussianThreshold`, `POTThreshold`, `create_threshold_strategy`.

---

### Files changed

| File                              | What changed                                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| `src/models/model.py`             | Learned decoder queries (full), tighter bottleneck (lightweight)                                |
| `src/inference/inference.py`      | Three threshold strategies + factory                                                            |
| `src/inference/__init__.py`       | New exports                                                                                     |
| `src/pipeline.py`                 | `detect()` threshold method support, `evaluate()` F1-optimal threshold, `MODEL_PRESETS` updated |
| `notebooks/nasa/inference.ipynb`  | Use `threshold_method="gaussian"`                                                               |
| `notebooks/power/inference.ipynb` | Use `threshold_method="gaussian"`                                                               |
| `README.md`                       | Documented new features                                                                         |
