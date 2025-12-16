"""
Inference module for anomaly detection.
Provides anomaly scoring and detection capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import json

from ..models.model import TransformerAutoencoder, LightweightTransformerAutoencoder
from ..data.data_loader import AnomalyDataLoader


class AnomalyDetector:
    """Anomaly detector using trained transformer autoencoder."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: AnomalyDataLoader,
        threshold_percentile: float = 95.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: trained transformer autoencoder
            data_loader: data loader used for training (for normalization)
            threshold_percentile: percentile for anomaly threshold
            device: device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.data_loader = data_loader
        self.device = device
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def calculate_reconstruction_errors(
        self, data: np.ndarray, window_size: int, stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reconstruction errors for data.

        Args:
            data: normalized time series data
            window_size: window size for sequences
            stride: stride for sliding window

        Returns:
            reconstruction errors and reconstructed data
        """
        errors = []
        reconstructions = []

        with torch.no_grad():
            for i in range(0, len(data) - window_size + 1, stride):
                window = data[i : i + window_size]
                window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)
                window_tensor = window_tensor.to(self.device)

                # Get reconstruction
                reconstructed = self.model(window_tensor)

                # Calculate error
                error = torch.abs(window_tensor - reconstructed)
                error = error.mean().cpu().numpy()

                errors.append(error)
                reconstructions.append(reconstructed.cpu().numpy())

        return np.array(errors), np.array(reconstructions)

    def fit_threshold(self, train_data: np.ndarray, window_size: int, stride: int = 1):
        """
        Fit anomaly threshold using training data.

        Args:
            train_data: normalized training data
            window_size: window size
            stride: stride
        """
        print("Calculating threshold from training data...")
        errors, _ = self.calculate_reconstruction_errors(
            train_data, window_size, stride
        )
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly threshold (p{self.threshold_percentile}): {self.threshold:.6f}")

        return self.threshold

    def detect_anomalies(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Detect anomalies in data.

        Args:
            data: raw time series data
            window_size: window size
            stride: stride
            threshold: custom threshold (uses fitted threshold if None)

        Returns:
            dictionary with anomaly information
        """
        # Normalize data
        normalized = self.data_loader.scaler.transform(data.reshape(-1, 1)).flatten()

        # Calculate errors
        errors, reconstructions = self.calculate_reconstruction_errors(
            normalized, window_size, stride
        )

        # Use threshold
        if threshold is None:
            if self.threshold is None:
                raise ValueError(
                    "No threshold set. Call fit_threshold first or provide threshold."
                )
            threshold = self.threshold

        # Detect anomalies
        anomaly_mask = errors > threshold

        # Map window-level anomalies back to point-level
        point_scores = np.zeros(len(data))
        point_counts = np.zeros(len(data))

        for i, (is_anomaly, error) in enumerate(zip(anomaly_mask, errors)):
            start_idx = i * stride
            end_idx = start_idx + window_size

            if is_anomaly:
                point_scores[start_idx:end_idx] += error
                point_counts[start_idx:end_idx] += 1

        # Average scores for points in multiple windows
        point_counts = np.maximum(point_counts, 1)  # Avoid division by zero
        point_scores = point_scores / point_counts

        # Points are anomalous if their average score exceeds threshold
        point_anomalies = point_scores > threshold

        # Find anomaly segments
        anomaly_segments = self._find_segments(point_anomalies)

        return {
            "point_scores": point_scores,
            "point_anomalies": point_anomalies,
            "window_errors": errors,
            "window_anomalies": anomaly_mask,
            "threshold": threshold,
            "n_anomalies": int(np.sum(point_anomalies)),
            "anomaly_rate": float(np.mean(point_anomalies)),
            "anomaly_segments": anomaly_segments,
        }

    def _find_segments(self, binary_mask: np.ndarray) -> List[Dict]:
        """Find continuous segments of anomalies."""
        segments = []
        in_segment = False
        start = 0

        for i, is_anomaly in enumerate(binary_mask):
            if is_anomaly and not in_segment:
                start = i
                in_segment = True
            elif not is_anomaly and in_segment:
                segments.append(
                    {"start": int(start), "end": int(i - 1), "length": int(i - start)}
                )
                in_segment = False

        # Handle case where anomaly extends to end
        if in_segment:
            segments.append(
                {
                    "start": int(start),
                    "end": int(len(binary_mask) - 1),
                    "length": int(len(binary_mask) - start),
                }
            )

        return segments

    def analyze_anomalies(self, results: Dict, data: np.ndarray) -> str:
        """Generate human-readable anomaly analysis."""
        lines = []
        lines.append("=" * 60)
        lines.append("ANOMALY DETECTION RESULTS")
        lines.append("=" * 60)
        lines.append(f"Total data points: {len(data)}")
        lines.append(f"Anomalous points: {results['n_anomalies']}")
        lines.append(f"Anomaly rate: {results['anomaly_rate'] * 100:.2f}%")
        lines.append(f"Threshold: {results['threshold']:.6f}")
        lines.append(
            f"\nNumber of anomaly segments: {len(results['anomaly_segments'])}"
        )

        if results["anomaly_segments"]:
            lines.append("\nTop anomaly segments:")
            # Sort by length
            sorted_segments = sorted(
                results["anomaly_segments"], key=lambda x: x["length"], reverse=True
            )[:5]

            for i, segment in enumerate(sorted_segments, 1):
                start, end, length = segment["start"], segment["end"], segment["length"]
                avg_score = results["point_scores"][start : end + 1].mean()
                lines.append(
                    f"  {i}. Indices {start}-{end} (length: {length}, avg_score: {avg_score:.4f})"
                )

        lines.append("=" * 60)

        return "\n".join(lines)

    def save_results(self, results: Dict, output_path: str):
        """Save detection results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        serializable = {
            "point_scores": results["point_scores"].tolist(),
            "point_anomalies": results["point_anomalies"].tolist(),
            "window_errors": results["window_errors"].tolist(),
            "window_anomalies": results["window_anomalies"].tolist(),
            "threshold": float(results["threshold"]),
            "n_anomalies": results["n_anomalies"],
            "anomaly_rate": results["anomaly_rate"],
            "anomaly_segments": results["anomaly_segments"],
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"Results saved to {output_path}")


def load_trained_model(
    checkpoint_path: str, model_type: str = "lightweight", device: str = None
) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: path to checkpoint file
        model_type: 'lightweight' or 'full'
        device: device to load model on

    Returns:
        loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    if model_type == "lightweight":
        model = LightweightTransformerAutoencoder(
            input_dim=1,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )
    else:
        model = TransformerAutoencoder(
            input_dim=1,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")

    return model


def detect_anomalies_from_file(
    data_path: str,
    checkpoint_path: str,
    model_type: str = "lightweight",
    window_size: int = 50,
    threshold_percentile: float = 95.0,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Complete pipeline for anomaly detection from file.

    Args:
        data_path: path to data JSON file
        checkpoint_path: path to model checkpoint
        model_type: model type
        window_size: window size
        threshold_percentile: threshold percentile
        output_path: path to save results (optional)

    Returns:
        detection results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading model...")
    model = load_trained_model(checkpoint_path, model_type, device)

    # Load data
    print("\nLoading data...")
    data_loader = AnomalyDataLoader(
        json_path=data_path,
        window_size=window_size,
        batch_size=32,
        train_split=0.8,
        normalize=True,
    )
    train_loader, val_loader, raw_data = data_loader.load_and_process()

    # Create detector
    print("\nInitializing detector...")
    detector = AnomalyDetector(
        model=model,
        data_loader=data_loader,
        threshold_percentile=threshold_percentile,
        device=device,
    )

    # Fit threshold on training data
    train_data = data_loader.processed_data[
        : int(len(data_loader.processed_data) * 0.8)
    ]
    detector.fit_threshold(train_data, window_size, stride=1)

    # Detect anomalies on full data
    print("\nDetecting anomalies...")
    results = detector.detect_anomalies(raw_data, window_size, stride=1)

    # Print analysis
    print("\n" + detector.analyze_anomalies(results, raw_data))

    # Save results if requested
    if output_path:
        detector.save_results(results, output_path)

    return results


if __name__ == "__main__":
    # Example usage
    results = detect_anomalies_from_file(
        data_path="./data-output.json",
        checkpoint_path="./checkpoints/best_model.pt",
        model_type="lightweight",
        window_size=50,
        threshold_percentile=95.0,
        output_path="./anomaly_results.json",
    )
