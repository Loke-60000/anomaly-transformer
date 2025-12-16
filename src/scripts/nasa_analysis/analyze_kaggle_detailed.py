#!/usr/bin/env python3

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import os
import json


def analyze_kaggle_data():
    print("DETAILED ANALYSIS OF NASA ANOMALY DETECTION DATASET")
    print("=" * 70)

    # Load the labeled anomalies
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
        "labeled_anomalies.csv",
    )

    print("Dataset Overview:")
    print(f"- Total number of channels: {len(df)}")
    print(f"- SMAP channels: {len(df[df['spacecraft'] == 'SMAP'])}")
    print(f"- MSL channels: {len(df[df['spacecraft'] == 'MSL'])}")
    print()

    # Parse anomaly sequences
    print("Anomaly Analysis:")
    total_anomalies = 0
    anomaly_lengths = []

    for idx, row in df.iterrows():
        chan_id = row["chan_id"]
        sequences = eval(row["anomaly_sequences"])  # Parse the string representation
        num_sequences = len(sequences)
        total_anomalies += num_sequences

        print(f"{chan_id} ({row['spacecraft']}): {num_sequences} anomaly sequences")
        for seq in sequences:
            length = seq[1] - seq[0]
            anomaly_lengths.append(length)
            print(f"  [{seq[0]}, {seq[1]}] - Length: {length}")

    print(f"\nTotal anomaly sequences across all channels: {total_anomalies}")
    print(f"Average anomaly length: {np.mean(anomaly_lengths):.1f}")
    print(f"Median anomaly length: {np.median(anomaly_lengths):.1f}")
    print(f"Min anomaly length: {min(anomaly_lengths)}")
    print(f"Max anomaly length: {max(anomaly_lengths)}")
    print()

    # Get dataset download path to access numpy files
    dataset_path = "/home/lokman/.cache/kagglehub/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/versions/1"

    # Try to load some actual time series data
    print("SAMPLE TIME SERIES DATA:")
    print("=" * 50)

    sample_channels = ["P-1", "S-1", "E-1"]
    for chan in sample_channels:
        # Try to load training data
        train_path = f"{dataset_path}/data/data/train/{chan}.npy"
        test_path = f"{dataset_path}/data/data/test/{chan}.npy"

        if os.path.exists(train_path):
            train_data = np.load(train_path)
            print(f"\n{chan} Training Data:")
            print(f"  Shape: {train_data.shape}")
            print(f"  Data type: {train_data.dtype}")
            print(f"  First 10 values: {train_data[:10]}")

        if os.path.exists(test_path):
            test_data = np.load(test_path)
            print(f"\n{chan} Test Data:")
            print(f"  Shape: {test_data.shape}")
            print(f"  Data type: {test_data.dtype}")
            print(f"  First 10 values: {test_data[:10]}")

    return df


if __name__ == "__main__":
    analyze_kaggle_data()
