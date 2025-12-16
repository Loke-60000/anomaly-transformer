#!/usr/bin/env python3

import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Tuple

# Try to import matplotlib, skip visualization if not available
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    print("Matplotlib not available, skipping visualizations")
    HAS_MATPLOTLIB = False


def create_nasa_data_loader():
    """
    Create a data loader for the NASA SMAP/MSL dataset.
    This will load the data, create proper train/test splits with labels.
    """

    print("CREATING NASA DATASET LOADER")
    print("=" * 50)

    # Load the labeled anomalies
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
        "labeled_anomalies.csv",
    )

    dataset_path = "/home/lokman/.cache/kagglehub/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/versions/1"

    print(f"Found {len(df)} channels")
    print(f"Spacecraft types: {df['spacecraft'].value_counts().to_dict()}")

    # Process each channel
    processed_data = {
        "train_data": [],
        "test_data": [],
        "train_labels": [],
        "test_labels": [],
        "channel_info": [],
    }

    for idx, row in df.iterrows():
        chan_id = row["chan_id"]
        spacecraft = row["spacecraft"]
        anomaly_sequences = eval(row["anomaly_sequences"])

        print(f"Processing {chan_id} ({spacecraft})...")

        # Load training and test data
        train_path = f"{dataset_path}/data/data/train/{chan_id}.npy"
        test_path = f"{dataset_path}/data/data/test/{chan_id}.npy"

        if os.path.exists(train_path) and os.path.exists(test_path):
            train_data = np.load(train_path)
            test_data = np.load(test_path)

            # Create labels for test data (training is assumed normal)
            test_labels = np.zeros(len(test_data))

            # Mark anomaly sequences in test data
            for start, end in anomaly_sequences:
                if end <= len(test_data):
                    test_labels[start:end] = 1
                else:
                    print(
                        f"Warning: Anomaly sequence {start}-{end} exceeds test data length {len(test_data)}"
                    )

            # Create training labels (all normal)
            train_labels = np.zeros(len(train_data))

            processed_data["train_data"].append(train_data)
            processed_data["test_data"].append(test_data)
            processed_data["train_labels"].append(train_labels)
            processed_data["test_labels"].append(test_labels)
            processed_data["channel_info"].append(
                {
                    "chan_id": chan_id,
                    "spacecraft": spacecraft,
                    "anomaly_sequences": anomaly_sequences,
                    "train_length": len(train_data),
                    "test_length": len(test_data),
                    "anomaly_ratio": test_labels.mean(),
                }
            )

            print(
                f"  Train: {train_data.shape}, Test: {test_data.shape}, Anomaly ratio: {test_labels.mean():.3f}"
            )
        else:
            print(f"  Warning: Data files not found for {chan_id}")

    print(f"\nSuccessfully processed {len(processed_data['train_data'])} channels")

    return processed_data


def create_training_sequences(
    data: np.ndarray, labels: np.ndarray, window_size: int = 50, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from time series data.

    Args:
        data: Time series data [timesteps, features]
        labels: Binary labels [timesteps]
        window_size: Size of sliding window
        stride: Stride for sliding window

    Returns:
        Tuple of (sequences, sequence_labels)
    """
    sequences = []
    sequence_labels = []

    for i in range(0, len(data) - window_size + 1, stride):
        window_data = data[i : i + window_size]
        window_labels = labels[i : i + window_size]

        # Label sequence as anomalous if any timestep in window is anomalous
        sequence_label = 1 if window_labels.sum() > 0 else 0

        sequences.append(window_data)
        sequence_labels.append(sequence_label)

    return np.array(sequences), np.array(sequence_labels)


def prepare_nasa_training_data(
    processed_data: Dict,
    window_size: int = 50,
    stride: int = 10,
    max_channels: int = None,
):
    """
    Prepare training data from NASA dataset for transformer training.

    Args:
        processed_data: Output from create_nasa_data_loader
        window_size: Sliding window size
        stride: Window stride
        max_channels: Maximum number of channels to use (for testing)

    Returns:
        Dictionary with prepared training data
    """

    print(f"\nPREPARING TRAINING DATA")
    print(f"Window size: {window_size}, Stride: {stride}")
    print("=" * 40)

    all_train_sequences = []
    all_train_labels = []
    all_test_sequences = []
    all_test_labels = []

    channels_to_process = min(
        len(processed_data["train_data"]),
        max_channels or len(processed_data["train_data"]),
    )

    for i in range(channels_to_process):
        train_data = processed_data["train_data"][i]
        test_data = processed_data["test_data"][i]
        train_labels = processed_data["train_labels"][i]
        test_labels = processed_data["test_labels"][i]
        channel_info = processed_data["channel_info"][i]

        print(f"Processing channel {channel_info['chan_id']}...")

        # Create sequences
        train_sequences, train_seq_labels = create_training_sequences(
            train_data, train_labels, window_size, stride
        )

        test_sequences, test_seq_labels = create_training_sequences(
            test_data, test_labels, window_size, stride
        )

        all_train_sequences.append(train_sequences)
        all_train_labels.append(train_seq_labels)
        all_test_sequences.append(test_sequences)
        all_test_labels.append(test_seq_labels)

        print(
            f"  Train sequences: {len(train_sequences)} ({train_seq_labels.sum()} anomalous)"
        )
        print(
            f"  Test sequences: {len(test_sequences)} ({test_seq_labels.sum()} anomalous)"
        )

    # Combine all channels
    combined_train_sequences = np.vstack(all_train_sequences)
    combined_train_labels = np.concatenate(all_train_labels)
    combined_test_sequences = np.vstack(all_test_sequences)
    combined_test_labels = np.concatenate(all_test_labels)

    print(f"\nCOMBINED DATASET STATISTICS:")
    print(
        f"Training sequences: {len(combined_train_sequences)} ({combined_train_labels.sum()} anomalous)"
    )
    print(
        f"Test sequences: {len(combined_test_sequences)} ({combined_test_labels.sum()} anomalous)"
    )
    print(f"Sequence shape: {combined_train_sequences.shape}")
    print(f"Features per timestep: {combined_train_sequences.shape[2]}")
    print(f"Training anomaly ratio: {combined_train_labels.mean():.4f}")
    print(f"Test anomaly ratio: {combined_test_labels.mean():.4f}")

    return {
        "train_sequences": combined_train_sequences,
        "train_labels": combined_train_labels,
        "test_sequences": combined_test_sequences,
        "test_labels": combined_test_labels,
        "window_size": window_size,
        "n_features": combined_train_sequences.shape[2],
        "n_channels": channels_to_process,
        "channel_info": processed_data["channel_info"][:channels_to_process],
    }


def save_processed_data(
    training_data: Dict, output_path: str = "nasa_processed_data.npz"
):
    """Save processed training data to file."""

    print(f"\nSaving processed data to {output_path}...")

    np.savez_compressed(
        output_path,
        train_sequences=training_data["train_sequences"],
        train_labels=training_data["train_labels"],
        test_sequences=training_data["test_sequences"],
        test_labels=training_data["test_labels"],
        window_size=training_data["window_size"],
        n_features=training_data["n_features"],
        n_channels=training_data["n_channels"],
    )

    # Save channel info as JSON
    info_path = output_path.replace(".npz", "_info.json")
    with open(info_path, "w") as f:
        json.dump(training_data["channel_info"], f, indent=2)

    print(f"Saved data to {output_path} and {info_path}")


def load_processed_data(data_path: str = "nasa_processed_data.npz") -> Dict:
    """Load processed training data from file."""

    print(f"Loading processed data from {data_path}...")

    data = np.load(data_path)

    # Load channel info
    info_path = data_path.replace(".npz", "_info.json")
    channel_info = []
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            channel_info = json.load(f)

    return {
        "train_sequences": data["train_sequences"],
        "train_labels": data["train_labels"],
        "test_sequences": data["test_sequences"],
        "test_labels": data["test_labels"],
        "window_size": int(data["window_size"]),
        "n_features": int(data["n_features"]),
        "n_channels": int(data["n_channels"]),
        "channel_info": channel_info,
    }


def visualize_sample_data(training_data: Dict, n_samples: int = 3):
    """Visualize some sample data."""

    print(f"\nVISUALIZING SAMPLE DATA")
    print("=" * 30)

    train_sequences = training_data["train_sequences"]
    train_labels = training_data["train_labels"]
    test_sequences = training_data["test_sequences"]
    test_labels = training_data["test_labels"]

    # Find some normal and anomalous sequences
    normal_indices = np.where(test_labels == 0)[0][:n_samples]
    anomalous_indices = np.where(test_labels == 1)[0][:n_samples]

    fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))
    fig.suptitle("NASA Dataset Sample Sequences")

    # Plot normal sequences
    for i, idx in enumerate(normal_indices):
        sequence = test_sequences[idx]
        axes[0, i].plot(sequence[:, 0], label="Feature 0", alpha=0.7)
        axes[0, i].plot(sequence[:, 1], label="Feature 1", alpha=0.7)
        axes[0, i].set_title(f"Normal Sequence {idx}")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

    # Plot anomalous sequences
    for i, idx in enumerate(anomalous_indices):
        sequence = test_sequences[idx]
        axes[1, i].plot(sequence[:, 0], label="Feature 0", alpha=0.7)
        axes[1, i].plot(sequence[:, 1], label="Feature 1", alpha=0.7)
        axes[1, i].set_title(f"Anomalous Sequence {idx}")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("nasa_sample_sequences.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved sample visualization to nasa_sample_sequences.png")

    # Print statistics
    print(f"Dataset statistics:")
    print(
        f"  Feature range: [{train_sequences.min():.3f}, {train_sequences.max():.3f}]"
    )
    print(f"  Feature std: {train_sequences.std(axis=(0, 1))}")
    print(f"  Normal sequences: {(train_labels == 0).sum() + (test_labels == 0).sum()}")
    print(
        f"  Anomalous sequences: {(train_labels == 1).sum() + (test_labels == 1).sum()}"
    )


def main():
    """Main function to process NASA dataset."""

    # Process the NASA dataset
    processed_data = create_nasa_data_loader()

    # Prepare training data (use first 10 channels for testing)
    training_data = prepare_nasa_training_data(
        processed_data,
        window_size=50,
        stride=5,
        max_channels=10,  # Start with subset for testing
    )

    # Save processed data
    save_processed_data(training_data, "nasa_processed_data.npz")

    # Visualize sample data
    try:
        visualize_sample_data(training_data)
    except Exception as e:
        print(f"Visualization failed: {e}")

    print(f"\nDATA PROCESSING COMPLETE!")
    print(f"Ready for transformer training with NASA dataset")
    print(f"Use the improved transformer architecture for best results")


if __name__ == "__main__":
    main()
