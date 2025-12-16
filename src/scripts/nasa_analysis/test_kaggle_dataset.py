#!/usr/bin/env python3

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import os


def main():
    print("Testing Kaggle NASA Anomaly Detection Dataset...")
    print("=" * 60)

    # Try to download and explore the dataset structure first
    try:
        # First, let's download the dataset to see what files are available
        print(
            "Downloading dataset: patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
        )

        # Download the dataset first to explore structure
        path = kagglehub.dataset_download(
            "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
        )
        print(f"Dataset downloaded to: {path}")

        # List all files in the dataset
        print("\nFiles in dataset:")
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                print(f"  {rel_path}")

        # Try to find CSV files specifically
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, path)
                    csv_files.append(rel_path)

        print(f"\nFound {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  {csv_file}")

        # Try to load the first CSV file if available
        if csv_files:
            first_csv = csv_files[0]
            print(f"\nTrying to load: {first_csv}")

            df = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
                first_csv,
            )

            print(f"Dataset loaded successfully!")
            print(f"Dataset type: {type(df)}")
            print(f"Dataset shape: {df.shape}")
            print()

            # Display basic information
            print("Dataset Info:")
            print(df.info())
            print()

            print("First 5 records:")
            print(df.head())
            print()

            print("Dataset columns:")
            print(df.columns.tolist())
            print()

            # Check for anomaly labels
            if (
                "label" in df.columns
                or "anomaly" in df.columns
                or "is_anomaly" in df.columns
            ):
                label_col = (
                    "label"
                    if "label" in df.columns
                    else ("anomaly" if "anomaly" in df.columns else "is_anomaly")
                )
                print(f"Found label column: '{label_col}'")
                print(f"Label distribution:")
                print(df[label_col].value_counts())
                print()

            # Look for other potential label columns
            potential_label_cols = [
                col
                for col in df.columns
                if any(
                    word in col.lower()
                    for word in ["label", "anomaly", "fault", "error", "normal"]
                )
            ]
            if potential_label_cols:
                print(f"Potential label columns found: {potential_label_cols}")
                for col in potential_label_cols:
                    print(f"  {col}: {df[col].unique()}")
                print()

            # Check data types
            print("Data types:")
            print(df.dtypes)
            print()

            # Check for missing values
            print("Missing values:")
            print(df.isnull().sum())
            print()

            # Statistical summary
            print("Statistical summary:")
            print(df.describe())

            # Try to load other CSV files too
            if len(csv_files) > 1:
                print(f"\n" + "=" * 60)
                print("EXPLORING OTHER CSV FILES:")
                print("=" * 60)

                for csv_file in csv_files[1:3]:  # Load up to 3 more files
                    try:
                        print(f"\nLoading: {csv_file}")
                        df_other = kagglehub.dataset_load(
                            KaggleDatasetAdapter.PANDAS,
                            "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
                            csv_file,
                        )
                        print(f"Shape: {df_other.shape}")
                        print(f"Columns: {df_other.columns.tolist()}")
                        print("Sample:")
                        print(df_other.head(3))
                        print()
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")

            return df
        else:
            print("No CSV files found in the dataset")
            return None

    except Exception as e:
        print(f"Error with dataset: {e}")
        print("This might be due to:")
        print("1. Network issues")
        print("2. Need for Kaggle authentication")
        print("3. Dataset access restrictions")

        print(f"\nError details: {type(e).__name__}: {str(e)}")
        return None


if __name__ == "__main__":
    df = main()
