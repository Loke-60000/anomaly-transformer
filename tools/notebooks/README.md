# Training Notebooks

This folder contains Jupyter notebooks for training and using the transformer anomaly detection system.

## Available Notebooks

### 1. `training_notebook.ipynb` - Complete Training Guide

**Full step-by-step training with detailed explanations.**

What's included:
- Data loading and validation
- Quality checks for your time series
- Configuration selection (presets + custom)
- Model training with progress tracking
- Training diagnostics (overfitting detection)
- Anomaly detection
- Results analysis
- Visualization generation
- Model saving and loading

**Use this when:** You want full control and understanding of the process.

**Time:** 10-20 minutes (including reading explanations)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   pip install jupyter notebook
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open the notebook** and follow the cells in order

4. **Replace data path** with your JSON file:
   ```python
   DATA_PATH = '../data-output.json'  # Change this
   ```

## Data Requirements

Your JSON file should have this structure:

```json
{
  "success": true,
  "data": {
    "timestamps": ["2024-01-01T00:00:00", ...],
    "values": [[[42.5, 0], [43.1, 0], ...]]
  }
}
```

- `timestamps`: ISO format timestamps (can be any frequency)
- `values`: Arrays of [value, label] pairs (labels are ignored)

## Expected Outputs

After running the notebook, you'll have:

```
transformeranomalygen/
├── results/
│   ├── time_series_with_anomalies.png
│   ├── reconstruction_error_distribution.png
│   ├── training_history.png
│   └── anomaly_results.json
├── checkpoints/
│   └── my_trained_model.pt
└── configs/
    └── my_config.json
```

## Configuration Presets

| Preset | Training Time | Use Case |
|--------|---------------|----------|
| `quick_test` | ~1 min | Testing, debugging |
| `balanced` | ~5 min | **Recommended default** |
| `fast_training` | ~3 min | Quick results needed |
| `high_accuracy` | ~15 min | Best detection quality |
| `production` | ~10 min | Production deployment |

## Common Issues

**Problem:** "Data too short" error  
**Solution:** Need at least 1,000 points (5,000+ recommended)

**Problem:** Training loss not decreasing  
**Solution:** Try `quick_test` preset first, check data normalization

**Problem:** Too many/few anomalies detected  
**Solution:** Adjust `threshold_percentile` (no retraining needed)

**Problem:** Out of memory  
**Solution:** Reduce `batch_size` or use `lightweight` model

## Tips

1. **Always start with `balanced` preset** - works well for most cases
2. **Check training curves** - both losses should decrease smoothly
3. **Iterate on threshold first** - cheaper than retraining
4. **Save successful configs** - reuse what works
5. **Your data can have anomalies** - model learns from majority pattern

## Need Help?

See the main README.md for:
- Detailed explanations of how the model works
- Hyperparameter tuning guide
- Troubleshooting section
- Performance characteristics
- Data quality requirements

## Example Usage Timeline

**First time (learning):**
1. Run `training_notebook.ipynb` cell-by-cell (~15 min)
2. Read explanations and understand outputs
3. Experiment with different configurations

**Regular use (production):**
1. Set your data path
2. Choose preset
3. Run "Quick Start" section (~5 min)
4. Review results
5. Adjust threshold if needed

## Next Steps

After training successfully:
- Review visualizations in `results/` folder
- Check if anomalies make sense for your domain
- Save model and config for reuse
- Set up batch processing for large datasets
- Consider deploying to production

Happy anomaly hunting! 🔍
