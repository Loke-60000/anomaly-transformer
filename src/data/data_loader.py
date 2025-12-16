"""
Data preprocessing utilities for time series anomaly detection.
Handles loading, normalization, and sequence preparation for transformer models.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class DataPreprocessor:
    """Handles data cleaning and preprocessing operations."""

    @staticmethod
    def clean_data(
        data: np.ndarray,
        remove_non_finite: bool = True,
        clip_outliers: bool = False,
        outlier_std_threshold: float = 4.0,
    ) -> np.ndarray:
        """
        Clean time series data by removing problematic values.

        Args:
            data: Input time series data
            remove_non_finite: Remove NaN and inf values
            clip_outliers: Clip extreme outliers
            outlier_std_threshold: Number of std deviations for outlier detection

        Returns:
            Cleaned data array
        """
        data = data.copy()

        # Remove non-finite values (NaN, inf)
        if remove_non_finite:
            finite_mask = np.isfinite(data)
            if not finite_mask.all():
                print(f"Warning: Removed {(~finite_mask).sum()} non-finite values")
                # Replace with interpolation or remove
                if finite_mask.sum() > 0:
                    # Replace with nearest valid value for now
                    valid_indices = np.where(finite_mask)[0]
                    for i in range(len(data)):
                        if not finite_mask[i]:
                            # Find nearest valid index
                            nearest = valid_indices[
                                np.argmin(np.abs(valid_indices - i))
                            ]
                            data[i] = data[nearest]

        # Clip extreme outliers
        if clip_outliers and len(data) > 0:
            mean = np.mean(data[np.isfinite(data)])
            std = np.std(data[np.isfinite(data)])
            if std > 0:
                lower_bound = mean - outlier_std_threshold * std
                upper_bound = mean + outlier_std_threshold * std

                outlier_mask = (data < lower_bound) | (data > upper_bound)
                n_outliers = outlier_mask.sum()
                if n_outliers > 0:
                    print(f"Warning: Clipped {n_outliers} extreme outliers")
                    data = np.clip(data, lower_bound, upper_bound)

        return data

    @staticmethod
    def validate_data(data: np.ndarray, min_length: int = 100) -> dict:
        """
        Validate data quality and return diagnostics.

        Args:
            data: Input time series data
            min_length: Minimum required length

        Returns:
            Dictionary with validation results and warnings
        """
        validation = {"valid": True, "warnings": [], "errors": [], "statistics": {}}

        # Check length
        if len(data) < min_length:
            validation["errors"].append(
                f"Data too short: {len(data)} points (minimum: {min_length})"
            )
            validation["valid"] = False

        # Check for non-finite values
        non_finite = ~np.isfinite(data)
        if non_finite.any():
            pct = 100 * non_finite.sum() / len(data)
            validation["warnings"].append(
                f"{non_finite.sum()} non-finite values ({pct:.1f}%)"
            )

        # Check for constant data
        if len(data) > 1:
            std = np.std(data[np.isfinite(data)])
            if std < 1e-10:
                validation["errors"].append("Data is constant (zero variance)")
                validation["valid"] = False
            elif std < 0.01:
                validation["warnings"].append("Data has very low variance")

        # Check for missing values (zeros)
        zeros = (data == 0).sum()
        if zeros > 0:
            pct = 100 * zeros / len(data)
            if pct > 30:
                validation["warnings"].append(
                    f"{zeros} zero values ({pct:.1f}% - may indicate missing data)"
                )

        # Statistics
        finite_data = data[np.isfinite(data)]
        if len(finite_data) > 0:
            validation["statistics"] = {
                "length": len(data),
                "finite_count": len(finite_data),
                "mean": float(np.mean(finite_data)),
                "std": float(np.std(finite_data)),
                "min": float(np.min(finite_data)),
                "max": float(np.max(finite_data)),
                "zeros": int(zeros),
            }

        return validation

    @staticmethod
    def interpolate_missing_values(
        data: np.ndarray, missing_values: list = [0, -1]
    ) -> np.ndarray:
        """
        Interpolate missing values in time series data.

        Args:
            data: Input time series data
            missing_values: List of values to treat as missing

        Returns:
            Data with interpolated values
        """
        data = data.copy()
        missing_mask = np.isin(data, missing_values)

        if not missing_mask.any():
            return data

        non_missing_indices = np.where(~missing_mask)[0]
        if len(non_missing_indices) == 0:
            return data

        first_valid = non_missing_indices[0]
        last_valid = non_missing_indices[-1]

        # Check for very long gaps
        max_gap = 0
        current_gap = 0
        for i in range(len(missing_mask)):
            if missing_mask[i]:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0

        if max_gap > 0.1 * len(data):
            print(
                f"Warning: Very long gap detected ({max_gap} consecutive missing values)"
            )

        # Interpolate middle values
        for i in range(first_valid, last_valid + 1):
            if missing_mask[i]:
                left_idx = DataPreprocessor._find_left_valid(
                    data, missing_mask, i, first_valid
                )
                right_idx = DataPreprocessor._find_right_valid(
                    data, missing_mask, i, last_valid
                )

                if left_idx >= first_valid and right_idx <= last_valid:
                    weight = (i - left_idx) / (right_idx - left_idx)
                    data[i] = data[left_idx] * (1 - weight) + data[right_idx] * weight

        # Forward fill leading values
        if first_valid > 0:
            data[:first_valid] = data[first_valid]

        # Backward fill trailing values
        if last_valid < len(data) - 1:
            data[last_valid + 1 :] = data[last_valid]

        return data

    @staticmethod
    def _find_left_valid(
        data: np.ndarray, mask: np.ndarray, idx: int, min_idx: int
    ) -> int:
        """Find nearest valid index to the left."""
        left_idx = idx - 1
        while left_idx >= min_idx and mask[left_idx]:
            left_idx -= 1
        return left_idx

    @staticmethod
    def _find_right_valid(
        data: np.ndarray, mask: np.ndarray, idx: int, max_idx: int
    ) -> int:
        """Find nearest valid index to the right."""
        right_idx = idx + 1
        while right_idx <= max_idx and mask[right_idx]:
            right_idx += 1
        return right_idx


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series with sliding window."""

    def __init__(self, data: np.ndarray, window_size: int = 50, stride: int = 1):
        """
        Args:
            data: Time series data of shape (n_samples,)
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.windows = self._create_windows()

    def _create_windows(self) -> np.ndarray:
        """Create sliding windows from time series."""
        num_windows = (len(self.data) - self.window_size) // self.stride + 1
        windows = np.array(
            [
                self.data[i * self.stride : i * self.stride + self.window_size]
                for i in range(num_windows)
            ]
        )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.windows[idx]
        return torch.FloatTensor(window).unsqueeze(-1)


class JSONDataLoader:
    """Loads time series data from JSON format."""

    @staticmethod
    def load_from_json(json_path: str) -> np.ndarray:
        """
        Load time series data from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Numpy array of time series values
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract values (first element of each [value, label] pair)
        values = data["data"]["values"][0]
        return np.array([v[0] for v in values])


class TimeSeriesNodeDataLoader:
    """Loads time series data from timeseries-data node format."""

    @staticmethod
    def load_from_node_json(json_path: str, unit_id: str = "73") -> np.ndarray:
        """
        Load time series data from node JSON file.

        Args:
            json_path: Path to node JSON file
            unit_id: Unit ID to extract (default "73" for temperature)

        Returns:
            Numpy array of time series values sorted by timestamp
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        # Navigate to the curve data
        curve_data = data["data"]["oCurveData"]["oData"]

        if unit_id not in curve_data:
            available_units = list(curve_data.keys())
            raise ValueError(
                f"Unit ID '{unit_id}' not found. Available units: {available_units}"
            )

        # Extract timestamp-value pairs
        measurements = curve_data[unit_id]["mResult"]

        # Sort by timestamp and extract values (first element of [value, status] pair)
        sorted_timestamps = sorted(measurements.keys(), key=int)
        values = [measurements[ts][0] for ts in sorted_timestamps]

        return np.array(values)

    @staticmethod
    def load_multiple_nodes(
        node_paths: list, unit_id: str = "73", combine_method: str = "concatenate"
    ) -> np.ndarray:
        """
        Load and combine data from multiple node files.

        Args:
            node_paths: List of paths to node JSON files
            unit_id: Unit ID to extract
            combine_method: How to combine data - "concatenate" or "average"

        Returns:
            Combined numpy array of time series values
        """
        all_data = []

        for path in node_paths:
            try:
                node_data = TimeSeriesNodeDataLoader.load_from_node_json(path, unit_id)
                all_data.append(node_data)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data could be loaded from any node files")

        if combine_method == "concatenate":
            return np.concatenate(all_data)
        elif combine_method == "average":
            # Average across nodes (requires same length)
            min_length = min(len(d) for d in all_data)
            trimmed_data = [d[:min_length] for d in all_data]
            return np.mean(trimmed_data, axis=0)
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")

    @staticmethod
    def load_from_index(
        index_path: str, unit_id: str = "73", combine_method: str = "concatenate"
    ) -> np.ndarray:
        """
        Load all successful nodes from index file.

        Args:
            index_path: Path to index.json file
            unit_id: Unit ID to extract
            combine_method: How to combine data - "concatenate" or "average"

        Returns:
            Combined numpy array of time series values
        """
        import os

        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Extract successful node file paths
        base_dir = os.path.dirname(index_path)
        node_paths = []

        for result in index_data["results"]:
            if result.get("bSuccess", False) and "sFilePath" in result:
                full_path = os.path.join(base_dir, result["sFilePath"])
                node_paths.append(full_path)

        print(f"Found {len(node_paths)} successful nodes in index")

        return TimeSeriesNodeDataLoader.load_multiple_nodes(
            node_paths, unit_id, combine_method
        )


class AnomalyDataLoader:
    """
    Main data loader for anomaly detection pipeline.
    Handles loading, preprocessing, normalization, and DataLoader creation.
    """

    def __init__(
        self,
        json_path: str,
        window_size: int = 50,
        stride: int = 1,
        batch_size: int = 32,
        train_split: float = 0.8,
        normalize: bool = True,
        data_source: str = "json",
        unit_id: str = "73",
        combine_method: str = "concatenate",
    ):
        """
        Args:
            json_path: Path to JSON data file
            window_size: Size of sliding window
            stride: Stride for sliding window
            batch_size: Batch size for DataLoader
            train_split: Proportion of data for training (0.0 to 1.0)
            normalize: Whether to normalize the data
            data_source: Type of data source - "json", "nodes", or "index"
            unit_id: Unit ID for node data (e.g., "73" for temperature)
            combine_method: How to combine multiple nodes - "concatenate" or "average"
        """
        self.json_path = json_path
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.train_split = train_split
        self.normalize = normalize
        self.data_source = data_source
        self.unit_id = unit_id
        self.combine_method = combine_method

        self.scaler = StandardScaler() if normalize else None
        self.raw_data = None
        self.processed_data = None

    def load_and_process(self) -> Tuple[DataLoader, DataLoader, np.ndarray]:
        """
        Load and process data, create train/val loaders.

        Returns:
            Tuple of (train_loader, val_loader, raw_data)
        """
        # Load raw data based on data source type
        if self.data_source == "json":
            self.raw_data = JSONDataLoader.load_from_json(self.json_path)
        elif self.data_source == "nodes":
            self.raw_data = TimeSeriesNodeDataLoader.load_from_node_json(
                self.json_path, self.unit_id
            )
        elif self.data_source == "index":
            self.raw_data = TimeSeriesNodeDataLoader.load_from_index(
                self.json_path, self.unit_id, self.combine_method
            )
        else:
            raise ValueError(
                f"Unknown data_source: {self.data_source}. Must be 'json', 'nodes', or 'index'"
            )

        # Preprocess
        self.processed_data = DataPreprocessor.interpolate_missing_values(
            self.raw_data.copy()
        )

        # Normalize
        if self.normalize:
            self.processed_data = self.scaler.fit_transform(
                self.processed_data.reshape(-1, 1)
            ).flatten()

        # Split data
        train_data, val_data = self._split_data()

        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, self.window_size, self.stride)
        val_dataset = TimeSeriesDataset(val_data, self.window_size, self.stride)

        # Create dataloaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)

        return train_loader, val_loader, self.raw_data

    def _split_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train and validation sets."""
        split_idx = int(len(self.processed_data) * self.train_split)
        train_data = self.processed_data[:split_idx]
        val_data = self.processed_data[split_idx:]
        return train_data, val_data

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create a PyTorch DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,  # Drop last batch only when shuffling (training)
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized data

        Returns:
            Data in original scale
        """
        if self.scaler is not None:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

    def get_statistics(self) -> dict:
        """
        Get descriptive statistics of the data.

        Returns:
            Dictionary with data statistics
        """
        if self.processed_data is None:
            return {}

        return {
            "n_samples": len(self.processed_data),
            "mean": float(np.mean(self.processed_data)),
            "std": float(np.std(self.processed_data)),
            "min": float(np.min(self.processed_data)),
            "max": float(np.max(self.processed_data)),
            "n_zeros": int(np.sum(self.raw_data == 0)),
            "n_negatives": int(np.sum(self.raw_data < 0)),
        }
