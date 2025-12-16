"""
Unit tests for data loading and preprocessing.
"""

import pytest
import numpy as np
import json
import torch
from pathlib import Path

import sys

sys.path.append("..")

from src.data.data_loader import (
    DataPreprocessor,
    TimeSeriesDataset,
    JSONDataLoader,
    AnomalyDataLoader,
)


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    def test_clean_data_no_issues(self):
        """Test cleaning data with no issues."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cleaned = DataPreprocessor.clean_data(data)
        np.testing.assert_array_equal(cleaned, data)

    def test_clean_data_with_nan(self):
        """Test cleaning data with NaN values."""
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        cleaned = DataPreprocessor.clean_data(data, remove_non_finite=True)
        assert np.isfinite(cleaned).all()

    def test_clean_data_with_inf(self):
        """Test cleaning data with inf values."""
        data = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        cleaned = DataPreprocessor.clean_data(data, remove_non_finite=True)
        assert np.isfinite(cleaned).all()

    def test_clean_data_clip_outliers(self):
        """Test outlier clipping."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # 100.0 is an outlier
        cleaned = DataPreprocessor.clean_data(
            data, clip_outliers=True, outlier_std_threshold=2.0
        )
        assert cleaned.max() < 100.0

    def test_validate_data_valid(self):
        """Test validation of valid data."""
        data = np.random.randn(1000)
        validation = DataPreprocessor.validate_data(data, min_length=100)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    def test_validate_data_too_short(self):
        """Test validation of data that's too short."""
        data = np.array([1.0, 2.0, 3.0])
        validation = DataPreprocessor.validate_data(data, min_length=100)
        assert validation["valid"] is False
        assert any("too short" in error.lower() for error in validation["errors"])

    def test_validate_data_constant(self):
        """Test validation of constant data."""
        data = np.ones(1000)
        validation = DataPreprocessor.validate_data(data, min_length=100)
        assert validation["valid"] is False
        assert any("constant" in error.lower() for error in validation["errors"])

    def test_validate_data_with_nan(self):
        """Test validation warns about NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 200)
        validation = DataPreprocessor.validate_data(data, min_length=100)
        assert any(
            "non-finite" in warning.lower() for warning in validation["warnings"]
        )

    def test_interpolate_missing_values_no_missing(self):
        """Test interpolation with no missing values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        interpolated = DataPreprocessor.interpolate_missing_values(
            data, missing_values=[0]
        )
        np.testing.assert_array_equal(interpolated, data)

    def test_interpolate_missing_values_with_zeros(self):
        """Test interpolation of zero values."""
        data = np.array([1.0, 0.0, 3.0, 0.0, 5.0])
        interpolated = DataPreprocessor.interpolate_missing_values(
            data, missing_values=[0]
        )

        # Check that zeros were interpolated
        assert interpolated[1] != 0.0
        assert interpolated[3] != 0.0

        # Check interpolation is reasonable (between neighbors)
        assert 1.0 < interpolated[1] < 3.0
        assert 3.0 < interpolated[3] < 5.0

    def test_interpolate_leading_trailing_missing(self):
        """Test interpolation of leading and trailing missing values."""
        data = np.array([0.0, 0.0, 2.0, 3.0, 0.0, 0.0])
        interpolated = DataPreprocessor.interpolate_missing_values(
            data, missing_values=[0]
        )

        # Leading values should be forward-filled
        assert interpolated[0] == 2.0
        assert interpolated[1] == 2.0

        # Trailing values should be backward-filled
        assert interpolated[4] == 3.0
        assert interpolated[5] == 3.0


class TestTimeSeriesDataset:
    """Test TimeSeriesDataset class."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        data = np.arange(100, dtype=float)
        dataset = TimeSeriesDataset(data, window_size=10, stride=1)

        assert len(dataset) > 0
        assert dataset.window_size == 10
        assert dataset.stride == 1

    def test_dataset_window_size(self):
        """Test dataset window creation."""
        data = np.arange(100, dtype=float)
        window_size = 10
        dataset = TimeSeriesDataset(data, window_size=window_size, stride=1)

        # Get first window
        window = dataset[0]
        assert isinstance(window, torch.Tensor)
        assert window.shape == (window_size, 1)

    def test_dataset_stride(self):
        """Test dataset stride."""
        data = np.arange(100, dtype=float)
        stride = 5
        dataset = TimeSeriesDataset(data, window_size=10, stride=stride)

        # With stride=5, we should have fewer windows
        expected_length = (len(data) - 10) // stride + 1
        assert len(dataset) == expected_length

    def test_dataset_indexing(self):
        """Test dataset indexing returns correct windows."""
        data = np.arange(20, dtype=float)
        dataset = TimeSeriesDataset(data, window_size=5, stride=1)

        # First window should be [0, 1, 2, 3, 4]
        window = dataset[0]
        expected = torch.FloatTensor([[0], [1], [2], [3], [4]])
        torch.testing.assert_close(window, expected)

        # Second window should be [1, 2, 3, 4, 5]
        window = dataset[1]
        expected = torch.FloatTensor([[1], [2], [3], [4], [5]])
        torch.testing.assert_close(window, expected)


class TestJSONDataLoader:
    """Test JSONDataLoader class."""

    def test_load_from_json(self, tmp_path):
        """Test loading data from JSON file."""
        # Create test JSON file
        test_data = {
            "data": {
                "values": [[[1.0, 0], [2.0, 0], [3.0, 0], [4.0, 0], [5.0, 0]]],
                "timestamps": ["t1", "t2", "t3", "t4", "t5"],
            }
        }

        json_file = tmp_path / "test.json"
        with open(json_file, "w") as f:
            json.dump(test_data, f)

        # Load data
        data = JSONDataLoader.load_from_json(str(json_file))

        assert isinstance(data, np.ndarray)
        assert len(data) == 5
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0, 4.0, 5.0])


class TestAnomalyDataLoader:
    """Test AnomalyDataLoader class."""

    @pytest.fixture
    def json_file(self, tmp_path):
        """Create a test JSON file."""
        data = {
            "data": {
                "values": [[[float(i), 0] for i in range(1000)]],
                "timestamps": [f"t{i}" for i in range(1000)],
            }
        }

        json_file = tmp_path / "test.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        return str(json_file)

    def test_data_loader_creation(self, json_file):
        """Test basic data loader creation."""
        loader = AnomalyDataLoader(
            json_path=json_file,
            window_size=50,
            batch_size=32,
            train_split=0.8,
            normalize=True,
        )

        assert loader.window_size == 50
        assert loader.batch_size == 32
        assert loader.train_split == 0.8
        assert loader.normalize is True

    def test_load_and_process(self, json_file):
        """Test loading and processing data."""
        loader = AnomalyDataLoader(
            json_path=json_file,
            window_size=50,
            batch_size=32,
            train_split=0.8,
            normalize=True,
        )

        train_loader, val_loader, raw_data = loader.load_and_process()

        # Check loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert raw_data is not None

        # Check data shapes
        assert len(raw_data) == 1000

    def test_train_val_split(self, json_file):
        """Test train/validation split."""
        loader = AnomalyDataLoader(
            json_path=json_file,
            window_size=50,
            batch_size=32,
            train_split=0.8,
            normalize=False,
        )

        train_loader, val_loader, _ = loader.load_and_process()

        # Train loader should have more batches than val loader
        assert len(train_loader) > len(val_loader)

    def test_normalization(self, json_file):
        """Test data normalization."""
        loader = AnomalyDataLoader(
            json_path=json_file,
            window_size=50,
            batch_size=32,
            train_split=0.8,
            normalize=True,
        )

        train_loader, val_loader, _ = loader.load_and_process()

        # Check scaler was fitted
        assert loader.scaler is not None
        assert hasattr(loader.scaler, "mean_")
        assert hasattr(loader.scaler, "scale_")

    def test_no_normalization(self, json_file):
        """Test without normalization."""
        loader = AnomalyDataLoader(
            json_path=json_file,
            window_size=50,
            batch_size=32,
            train_split=0.8,
            normalize=False,
        )

        train_loader, val_loader, _ = loader.load_and_process()

        # Scaler should be None
        assert loader.scaler is None

    def test_inverse_transform(self, json_file):
        """Test inverse transformation."""
        loader = AnomalyDataLoader(
            json_path=json_file,
            window_size=50,
            batch_size=32,
            train_split=0.8,
            normalize=True,
        )

        train_loader, val_loader, raw_data = loader.load_and_process()

        # Get normalized data
        normalized = loader.processed_data[:10]

        # Inverse transform
        denormalized = loader.inverse_transform(normalized)

        # Should be close to original (first 10 points)
        np.testing.assert_array_almost_equal(denormalized, raw_data[:10], decimal=5)

    def test_get_statistics(self, json_file):
        """Test getting data statistics."""
        loader = AnomalyDataLoader(
            json_path=json_file,
            window_size=50,
            batch_size=32,
            train_split=0.8,
            normalize=True,
        )

        train_loader, val_loader, _ = loader.load_and_process()
        stats = loader.get_statistics()

        assert "n_samples" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["n_samples"] == 1000
