"""convert_timeseries_to_output.py

Convert node JSON files to the legacy `data-output.json` format expected by
the baseline pipeline.

Default input directory in this repo:
    assets/data/timeseries-data/nodes/

Usage:
    python src/scripts/timeseries/convert_timeseries_to_output.py
    python src/scripts/timeseries/convert_timeseries_to_output.py --unit-id 73 --combine concatenate

Then:
    python run.py --preset balanced
"""

import json
from pathlib import Path
import numpy as np
import argparse


def load_node_data(json_path: str, unit_id: str = "73") -> list:
    """
    Load time series data from a single node JSON file.

    Args:
        json_path: Path to node JSON file
        unit_id: Unit ID to extract (default "73" for temperature)

    Returns:
        List of [value, 0] pairs
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Navigate to the curve data
    curve_data = data["data"]["oCurveData"]["oData"]

    if unit_id not in curve_data:
        available_units = list(curve_data.keys())
        raise ValueError(
            f"Unit ID '{unit_id}' not found in {json_path}. Available units: {available_units}"
        )

    # Extract timestamp-value pairs
    measurements = curve_data[unit_id]["mResult"]

    # Sort by timestamp and extract values (first element of [value, status] pair)
    sorted_timestamps = sorted(measurements.keys(), key=int)
    values = [[measurements[ts][0], 0] for ts in sorted_timestamps]  # [value, label=0]

    return values


def combine_all_nodes(
    nodes_dir: str, unit_id: str = "73", combine_method: str = "concatenate"
) -> list:
    """
    Combine all node files into a single dataset.

    Args:
        nodes_dir: Directory containing node JSON files
        unit_id: Unit ID to extract
        combine_method: "concatenate" or "average"

    Returns:
        List of [value, label] pairs
    """
    nodes_path = Path(nodes_dir)
    node_files = sorted(nodes_path.glob("*.json"))

    print(f"Found {len(node_files)} node files:")

    all_data = []
    successful = 0

    for node_file in node_files:
        try:
            node_data = load_node_data(str(node_file), unit_id)
            all_data.append(node_data)
            print(f"  ✓ {node_file.name}: {len(node_data)} points")
            successful += 1
        except Exception as e:
            print(f"  ✗ {node_file.name}: {str(e)}")
            continue

    if not all_data:
        raise ValueError("No data could be loaded from any node files")

    print(f"\nSuccessfully loaded {successful}/{len(node_files)} nodes")

    if combine_method == "concatenate":
        # Concatenate all data sequentially
        combined = []
        for node_data in all_data:
            combined.extend(node_data)
        print(f"Combined method: concatenate")
        print(f"Total data points: {len(combined)}")
        return combined

    elif combine_method == "average":
        # Average across nodes (requires same length)
        min_length = min(len(d) for d in all_data)
        print(
            f"Combined method: average (using first {min_length} points from each node)"
        )

        # Trim all to same length
        trimmed_data = [d[:min_length] for d in all_data]

        # Average values
        averaged = []
        for i in range(min_length):
            values = [d[i][0] for d in trimmed_data]
            avg_value = np.mean(values)
            averaged.append([avg_value, 0])

        print(f"Total data points: {len(averaged)}")
        return averaged
    else:
        raise ValueError(f"Unknown combine_method: {combine_method}")


def create_output_json(combined_data: list, output_path: str = "data-output.json"):
    """
    Create data-output.json in the format expected by the pipeline.

    Args:
        combined_data: List of [value, label] pairs
        output_path: Output file path
    """
    output_data = {
        "data": {"values": [combined_data]},
        "metadata": {
            "source": "timeseries-data/nodes/",
            "total_points": len(combined_data),
            "description": "Combined data from multiple node files",
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Created {output_path}")
    print(f"  Total data points: {len(combined_data)}")


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert timeseries node JSON files to data-output.json"
    )
    parser.add_argument(
        "--nodes-dir",
        default="assets/data/timeseries-data/nodes",
        help="Directory containing node JSON files (default: assets/data/timeseries-data/nodes)",
    )
    parser.add_argument(
        "--output",
        default="data-output.json",
        help="Output file path (default: data-output.json)",
    )
    parser.add_argument(
        "--unit-id",
        default="73",
        help="Unit ID to extract (default: 73)",
    )
    parser.add_argument(
        "--combine",
        choices=["concatenate", "average"],
        default="concatenate",
        help="How to combine multiple nodes (default: concatenate)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Converting timeseries node JSON files to data-output.json format")
    print("=" * 70)
    print()

    # Check if nodes directory exists
    nodes_dir = Path(args.nodes_dir)
    if not nodes_dir.exists():
        print(f"Error: Directory '{nodes_dir}' not found!")
        print("Tip: run this from the project root, or pass --nodes-dir")
        return

    # Combine all nodes
    combined_data = combine_all_nodes(str(nodes_dir), args.unit_id, args.combine)

    # Create output file
    create_output_json(combined_data, args.output)

    # Show statistics
    values = [d[0] for d in combined_data]
    print(f"\nData Statistics:")
    print(f"  Min: {min(values):.2f}")
    print(f"  Max: {max(values):.2f}")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Std: {np.std(values):.2f}")

    print("\n" + "=" * 70)
    print("✓ Conversion complete!")
    print("=" * 70)
    print("\nNow you can run the baseline pipeline:")
    print(f"  python run.py --data-path {args.output} --preset balanced")
    print()


if __name__ == "__main__":
    main()
