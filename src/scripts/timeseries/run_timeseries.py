"""
Run anomaly detection on timeseries-data (node format).

This script demonstrates different ways to use the new timeseries-data format:
1. Single node: Train on one node's data
2. Multiple nodes (concatenate): Combine all nodes sequentially
3. Multiple nodes (average): Average across nodes

Usage:
    # Single node
    python run_timeseries.py --mode single --node-file timeseries-data/nodes/2672_Hamburg_(EDDH).json

    # All nodes concatenated
    python run_timeseries.py --mode index --index-file timeseries-data/index.json

    # All nodes averaged
    python run_timeseries.py --mode index --index-file timeseries-data/index.json --combine average
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import AnomalyDetectionPipeline
from src.config.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    VisualizationConfig,
    PipelineConfig,
)


def run_single_node_example():
    """Example: Train on a single node (Hamburg airport temperature)."""
    print("=" * 70)
    print("SINGLE NODE EXAMPLE: Hamburg Airport Temperature")
    print("=" * 70)
    print()

    # Note: 169 data points is very small, so we reduce window_size and batch_size
    config = PipelineConfig(
        data=DataConfig(
            json_path="timeseries-data/nodes/2672_Hamburg_(EDDH).json",
            data_source="nodes",  # Use node format
            unit_id="73",  # Temperature unit
            window_size=20,  # Smaller window for small dataset
            batch_size=8,
            train_split=0.7,  # 70% train, 30% validation
        ),
        model=ModelConfig(
            model_type="lightweight",
            d_model=16,  # Smaller model for small dataset
            nhead=2,
            num_layers=1,
        ),
        training=TrainingConfig(
            epochs=50,
            learning_rate=1e-3,
            early_stopping_patience=10,
            checkpoint_dir=Path("./checkpoints_single_node"),
        ),
        inference=InferenceConfig(
            threshold_percentile=95.0,
            checkpoint_path=Path("./checkpoints_single_node/best_model.pt"),
            output_dir=Path("./results_single_node"),
        ),
        visualization=VisualizationConfig(output_dir=Path("./plots_single_node")),
    )

    # Create and run pipeline
    pipeline = AnomalyDetectionPipeline(config)
    results = pipeline.run_full_pipeline()

    print()
    print("=" * 70)
    print("✓ Single node training complete!")
    print(f"  Results: ./results_single_node/")
    print(f"  Plots: ./plots_single_node/")
    print("=" * 70)

    return results


def run_index_example(combine_method="concatenate"):
    """Example: Train on all nodes from index."""
    print("=" * 70)
    print(f"INDEX EXAMPLE: All Nodes ({combine_method.upper()})")
    print("=" * 70)
    print()

    # For concatenated data: 12 nodes × 169 points = 2028 points (more reasonable)
    config = PipelineConfig(
        data=DataConfig(
            json_path="timeseries-data/index.json",
            data_source="index",  # Use index to load all successful nodes
            unit_id="73",
            combine_method=combine_method,  # "concatenate" or "average"
            window_size=30 if combine_method == "concatenate" else 20,
            batch_size=16 if combine_method == "concatenate" else 8,
            train_split=0.8,
        ),
        model=ModelConfig(
            model_type="lightweight",
            d_model=32,
            nhead=2,
            num_layers=2,
        ),
        training=TrainingConfig(
            epochs=100,
            learning_rate=1e-3,
            early_stopping_patience=15,
            checkpoint_dir=Path(f"./checkpoints_index_{combine_method}"),
        ),
        inference=InferenceConfig(
            threshold_percentile=95.0,
            checkpoint_path=Path(f"./checkpoints_index_{combine_method}/best_model.pt"),
            output_dir=Path(f"./results_index_{combine_method}"),
        ),
        visualization=VisualizationConfig(
            output_dir=Path(f"./plots_index_{combine_method}")
        ),
    )

    # Create and run pipeline
    pipeline = AnomalyDetectionPipeline(config)
    results = pipeline.run_full_pipeline()

    print()
    print("=" * 70)
    print(f"✓ Index ({combine_method}) training complete!")
    print(f"  Results: ./results_index_{combine_method}/")
    print(f"  Plots: ./plots_index_{combine_method}/")
    print("=" * 70)

    return results


def main():
    """Main entry point with CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run anomaly detection on timeseries-data"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "index"],
        default="index",
        help="Run mode: single node or index (all nodes)",
    )
    parser.add_argument(
        "--node-file",
        default="timeseries-data/nodes/2672_Hamburg_(EDDH).json",
        help="Path to node JSON file (for single mode)",
    )
    parser.add_argument(
        "--index-file",
        default="timeseries-data/index.json",
        help="Path to index JSON file (for index mode)",
    )
    parser.add_argument(
        "--combine",
        choices=["concatenate", "average"],
        default="concatenate",
        help="How to combine multiple nodes (for index mode)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Transformer-Based Anomaly Detection")
    print("Using timeseries-data format")
    print("=" * 70)
    print()

    if args.mode == "single":
        run_single_node_example()
    elif args.mode == "index":
        run_index_example(combine_method=args.combine)


if __name__ == "__main__":
    # Quick start: run index mode with concatenation (recommended)
    # For CLI options, uncomment the line below
    # main()

    # Default: run index mode (all nodes concatenated)
    print("Quick start: Running index mode with concatenation")
    print("For more options, see the script comments or use --help")
    print()

    run_index_example(combine_method="concatenate")
