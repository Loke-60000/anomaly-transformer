"""run.py

Simple entry point for running anomaly detection using the *baseline* (non-improved)
transformer models wired through the OOP pipeline.

Usage:
    python run.py
    python run.py --preset quick_test
    python run.py --data-path ./data-output.json --preset balanced
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import AnomalyDetectionApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/run baseline transformer anomaly detection pipeline"
    )
    parser.add_argument(
        "--data-path",
        default="./data-output.json",
        help="Path to input data file (default: ./data-output.json)",
    )
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=[
            "quick_test",
            "balanced",
            "high_accuracy",
            "fast_training",
            "production",
        ],
        help="Pipeline preset (baseline model). Default: balanced",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Transformer-Based Anomaly Detection")
    print("=" * 70)
    print()

    # Create app
    app = AnomalyDetectionApp(data_path=args.data_path)

    print(f"Running with '{args.preset}' preset...")
    if args.preset == "quick_test":
        print("This should take ~1 minute.")
    elif args.preset == "balanced":
        print("This will typically take a few minutes.")
    elif args.preset == "high_accuracy":
        print("This can take longer (larger baseline transformer).")
    print()

    results = app.run_with_preset(args.preset)

    print()
    print("=" * 70)
    print("✓ Complete! Check the 'results' folder for visualizations.")
    print("=" * 70)
