# Scripts Directory

This directory contains various utility and analysis scripts for the transformer anomaly detection project.

## Directory Structure

### `nasa_analysis/`

Scripts for analyzing and comparing the NASA SMAP/MSL dataset:

- `analyze_kaggle_detailed.py` - Detailed analysis of NASA dataset structure
- `compare_datasets.py` - Comparison between NASA and local timeseries data
- `compare_datasets_simple.py` - Simplified dataset comparison
- `test_kaggle_dataset.py` - Initial NASA dataset exploration

### `data_processing/`

Scripts for data preparation and processing:

- `prepare_nasa_data.py` - Process NASA dataset for transformer training

### `timeseries/`

Scripts for timeseries data processing:

- `convert_timeseries_to_output.py` - Convert timeseries data to output format
- `run_timeseries.py` - Run timeseries analysis

## Usage

Run scripts from the project root directory:

```bash
# Process NASA data
python scripts/data_processing/prepare_nasa_data.py

# Train (CPU quick run)
python src/scripts/train_dsa_cpu.py --device cpu --epochs 5  --eval   # tiny sanity
python src/scripts/train_dsa_cpu.py --device cpu --epochs 50 --eval   # longer CPU run

# Train (full/GPU-ready, presets; default model-type is full)
#   --pico  : tiny preset
#   --ookii : large preset
python src/scripts/train_nasa_full.py --epochs 100 --eval
python src/scripts/train_nasa_full.py --pico  --eval        # tiny quick check
python src/scripts/train_nasa_full.py --ookii --eval        # big/full run

# Compare datasets
python scripts/nasa_analysis/compare_datasets_simple.py

# Run timeseries processing
python scripts/timeseries/run_timeseries.py
```
