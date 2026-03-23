import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import json
from tqdm.auto import tqdm

from ..models.model import (
    BaseAutoencoder,
    TransformerAutoencoder,
    LightweightTransformerAutoencoder,
)
from ..data.data_loader import AnomalyDataLoader


# ---------------------------------------------------------------------------
# Thresholding strategies
# ---------------------------------------------------------------------------

class ThresholdStrategy:
    """Base class — override ``fit`` and ``threshold`` property."""

    def fit(self, errors: np.ndarray) -> "ThresholdStrategy":
        raise NotImplementedError

    @property
    def threshold(self) -> float:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        raise NotImplementedError


class PercentileThreshold(ThresholdStrategy):
    """Original simple percentile (kept for backwards compatibility)."""

    def __init__(self, percentile: float = 95.0):
        self.percentile = percentile
        self._threshold: Optional[float] = None

    def fit(self, errors: np.ndarray) -> "PercentileThreshold":
        self._threshold = float(np.percentile(errors, self.percentile))
        return self

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise ValueError("Call fit() first")
        return self._threshold

    def to_dict(self) -> Dict:
        return {"method": "percentile", "percentile": self.percentile,
                "threshold": self._threshold}


class GaussianThreshold(ThresholdStrategy):
    """Fit N(mu, sigma) to training errors; threshold = mu + k * sigma.

    Much more principled than a fixed percentile: adapts to the actual
    error distribution shape and is independent of dataset size.
    ``k=3`` ≈ 99.87th percentile for a true Gaussian; ``k=4`` is stricter.
    """

    def __init__(self, k: float = 3.0):
        self.k = k
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None
        self._threshold: Optional[float] = None

    def fit(self, errors: np.ndarray) -> "GaussianThreshold":
        self._mu = float(np.mean(errors))
        self._sigma = float(np.std(errors))
        self._threshold = self._mu + self.k * self._sigma
        return self

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise ValueError("Call fit() first")
        return self._threshold

    def to_dict(self) -> Dict:
        return {"method": "gaussian", "k": self.k,
                "mu": self._mu, "sigma": self._sigma,
                "threshold": self._threshold}


class POTThreshold(ThresholdStrategy):
    """Peaks Over Threshold — extreme-value-theory-based adaptive threshold.

    1. Pick an initial ``q``-quantile as the "high" watermark.
    2. Collect exceedances above that watermark.
    3. Fit a Generalized Pareto Distribution (GPD) to the exceedances.
    4. Extrapolate the tail to find the threshold at the desired risk
       level ``risk_level`` (lower = stricter, e.g. 1e-4).

    This is the gold standard for anomaly thresholding — it adapts to
    heavy-tailed error distributions where Gaussian assumptions fail.
    """

    def __init__(self, risk_level: float = 1e-4, q: float = 0.98):
        self.risk_level = risk_level
        self.q = q
        self._threshold: Optional[float] = None

    def fit(self, errors: np.ndarray) -> "POTThreshold":
        from scipy.stats import genpareto

        t0 = float(np.quantile(errors, self.q))
        exceedances = errors[errors > t0] - t0

        if len(exceedances) < 10:
            # Not enough tail data — fall back to Gaussian
            mu, sigma = float(np.mean(errors)), float(np.std(errors))
            self._threshold = mu + 4.0 * sigma
            self._method_used = "gaussian_fallback"
            return self

        # MLE fit of GPD
        shape, _, scale = genpareto.fit(exceedances, floc=0)

        # Inverse survival function of the GPD, shifted back
        n = len(errors)
        n_exceed = len(exceedances)
        if shape == 0:
            self._threshold = t0 + scale * np.log(n_exceed / (n * self.risk_level))
        else:
            self._threshold = t0 + (scale / shape) * (
                (n_exceed / (n * self.risk_level)) ** shape - 1
            )
        self._threshold = float(self._threshold)
        self._method_used = "pot"
        return self

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise ValueError("Call fit() first")
        return self._threshold

    def to_dict(self) -> Dict:
        return {"method": "pot", "risk_level": self.risk_level,
                "q": self.q, "threshold": self._threshold}


def create_threshold_strategy(
    method: str = "gaussian", **kwargs
) -> ThresholdStrategy:
    """Factory for threshold strategies.

    Parameters
    ----------
    method : str
        ``"percentile"`` — fixed percentile (default 95).
        ``"gaussian"``   — mean + k * sigma (default k=3).
        ``"pot"``        — Peaks Over Threshold / EVT (default risk=1e-4).
    """
    strategies = {
        "percentile": PercentileThreshold,
        "gaussian": GaussianThreshold,
        "pot": POTThreshold,
    }
    if method not in strategies:
        raise ValueError(f"Unknown threshold method '{method}'. "
                         f"Choose from {list(strategies)}")
    return strategies[method](**kwargs)


class AnomalyDetector:
    def __init__(
        self,
        model: nn.Module,
        data_loader: AnomalyDataLoader,
        threshold_percentile: float = 95.0,
        threshold_method: str = "gaussian",
        threshold_kwargs: Optional[Dict] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.data_loader = data_loader
        self.device = device
        self.threshold_percentile = threshold_percentile

        # Build threshold strategy
        if threshold_kwargs is None:
            threshold_kwargs = {}
        if threshold_method == "percentile" and "percentile" not in threshold_kwargs:
            threshold_kwargs["percentile"] = threshold_percentile
        self._threshold_strategy = create_threshold_strategy(
            threshold_method, **threshold_kwargs
        )
        self.threshold = None

    def calculate_reconstruction_errors(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        errors = []
        reconstructions = []
        pre_windowed = data.ndim == 3 and data.shape[1] == window_size

        if pre_windowed:
            windows = data
        else:
            windows = np.array(
                [
                    data[i : i + window_size]
                    for i in range(0, len(data) - window_size + 1, stride)
                ],
                dtype=np.float32,
            )
            if windows.ndim == 2:
                windows = windows[..., np.newaxis]

        n_windows = len(windows)
        n_batches = (n_windows + batch_size - 1) // batch_size
        with torch.no_grad():
            for start in tqdm(
                range(0, n_windows, batch_size),
                total=n_batches,
                desc="Scoring windows",
                unit="batch",
            ):
                batch = windows[start : start + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)

                reconstructed = self.model(batch_tensor)
                batch_errors = (
                    torch.abs(batch_tensor - reconstructed)
                    .mean(dim=(1, 2))
                    .cpu()
                    .numpy()
                )
                reconstructions.append(reconstructed.cpu().numpy())
                errors.append(batch_errors)

        errors = np.concatenate(errors)
        reconstructions = (
            np.concatenate(reconstructions) if reconstructions else np.array([])
        )
        return errors, reconstructions

    def extract_feature_vectors(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        batch_size: int = 64,
        pooling: str = "mean",
    ) -> np.ndarray:
        if not isinstance(self.model, BaseAutoencoder):
            raise TypeError(
                "Feature extraction requires a model implementing BaseAutoencoder.encode()."
            )

        vectors = []
        pre_windowed = data.ndim == 3 and data.shape[1] == window_size

        if pre_windowed:
            windows = data
        else:
            windows = np.array(
                [
                    data[i : i + window_size]
                    for i in range(0, len(data) - window_size + 1, stride)
                ],
                dtype=np.float32,
            )
            if windows.ndim == 2:
                windows = windows[..., np.newaxis]

        n_windows = len(windows)
        n_batches = (n_windows + batch_size - 1) // batch_size

        with torch.no_grad():
            for start in tqdm(
                range(0, n_windows, batch_size),
                total=n_batches,
                desc="Extracting vectors",
                unit="batch",
            ):
                batch = windows[start : start + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)

                encoded = self.model.encode(batch_tensor)

                if pooling == "mean":
                    pooled = encoded.mean(dim=1)
                elif pooling == "max":
                    pooled = encoded.max(dim=1).values
                elif pooling == "last":
                    pooled = encoded[:, -1, :]
                elif pooling == "flatten":
                    pooled = encoded.reshape(encoded.shape[0], -1)
                else:
                    raise ValueError(
                        "Invalid pooling strategy. Choose from 'mean', 'max', 'last', 'flatten'."
                    )

                vectors.append(pooled.cpu().numpy())

        return np.concatenate(vectors) if vectors else np.array([])

    def save_feature_vectors(self, vectors: np.ndarray, output_path: str):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.suffix == ".npz":
            np.savez_compressed(output, vectors=vectors)
        else:
            np.save(output, vectors)

        print(f"Feature vectors saved to {output}")

    def fit_threshold(self, train_data: np.ndarray, window_size: int, stride: int = 1):
        print("Calculating threshold from training data...")
        errors, _ = self.calculate_reconstruction_errors(
            train_data, window_size, stride
        )
        self._threshold_strategy.fit(errors)
        self.threshold = self._threshold_strategy.threshold
        info = self._threshold_strategy.to_dict()
        print(f"Anomaly threshold ({info['method']}): {self.threshold:.6f}")

        return self.threshold

    def detect_anomalies(
        self,
        data: np.ndarray,
        window_size: int,
        stride: int = 1,
        threshold: Optional[float] = None,
    ) -> Dict:
        normalized = self.data_loader.scaler.transform(data.reshape(-1, 1)).flatten()

        errors, reconstructions = self.calculate_reconstruction_errors(
            normalized, window_size, stride
        )

        if threshold is None:
            if self.threshold is None:
                raise ValueError(
                    "No threshold set. Call fit_threshold first or provide threshold."
                )
            threshold = self.threshold

        anomaly_mask = errors > threshold

        point_scores = np.zeros(len(data))
        point_counts = np.zeros(len(data))

        for i, (is_anomaly, error) in enumerate(zip(anomaly_mask, errors)):
            start_idx = i * stride
            end_idx = start_idx + window_size

            if is_anomaly:
                point_scores[start_idx:end_idx] += error
                point_counts[start_idx:end_idx] += 1

        point_counts = np.maximum(point_counts, 1)
        point_scores = point_scores / point_counts
        point_anomalies = point_scores > threshold

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

    @staticmethod
    def load_model(
        checkpoint_path: str, model_type: str = "lightweight", device: str = None
    ) -> nn.Module:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

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

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Loaded {model_type} model from {checkpoint_path}")
        print(f"Training epoch: {checkpoint['epoch']}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")

        return model

    @classmethod
    def from_file(
        cls,
        data_path: str,
        checkpoint_path: str,
        model_type: str = "lightweight",
        window_size: int = 50,
        threshold_percentile: float = 95.0,
        output_path: Optional[str] = None,
    ) -> Dict:
        """Complete pipeline: load model + data, detect anomalies, optionally save results."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading model...")
        model = cls.load_model(checkpoint_path, model_type, device)

        print("\nLoading data...")
        data_loader = AnomalyDataLoader(
            json_path=data_path,
            window_size=window_size,
            batch_size=32,
            train_split=0.8,
            normalize=True,
        )
        train_loader, val_loader, raw_data = data_loader.load_and_process()

        print("\nInitializing detector...")
        detector = cls(
            model=model,
            data_loader=data_loader,
            threshold_percentile=threshold_percentile,
            device=device,
        )

        train_data = data_loader.processed_data[
            : int(len(data_loader.processed_data) * 0.8)
        ]
        detector.fit_threshold(train_data, window_size, stride=1)

        print("\nDetecting anomalies...")
        results = detector.detect_anomalies(raw_data, window_size, stride=1)

        print("\n" + detector.analyze_anomalies(results, raw_data))

        if output_path:
            detector.save_results(results, output_path)

        return results


# Backwards-compatible aliases
def load_trained_model(
    checkpoint_path: str, model_type: str = "lightweight", device: str = None
) -> nn.Module:
    return AnomalyDetector.load_model(checkpoint_path, model_type, device)


def detect_anomalies_from_file(
    data_path: str,
    checkpoint_path: str,
    model_type: str = "lightweight",
    window_size: int = 50,
    threshold_percentile: float = 95.0,
    output_path: Optional[str] = None,
) -> Dict:
    return AnomalyDetector.from_file(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        window_size=window_size,
        threshold_percentile=threshold_percentile,
        output_path=output_path,
    )


def extract_vectors_from_file(
    data_path: str,
    checkpoint_path: str,
    model_type: str = "lightweight",
    window_size: int = 50,
    stride: int = 1,
    batch_size: int = 64,
    pooling: str = "mean",
    output_path: Optional[str] = None,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = AnomalyDetector.load_model(checkpoint_path, model_type, device)

    print("\nLoading data...")
    data_loader = AnomalyDataLoader(
        json_path=data_path,
        window_size=window_size,
        batch_size=32,
        train_split=0.8,
        normalize=True,
    )
    _, _, raw_data = data_loader.load_and_process()

    detector = AnomalyDetector(
        model=model,
        data_loader=data_loader,
        threshold_percentile=95.0,
        device=device,
    )

    normalized = data_loader.scaler.transform(raw_data.reshape(-1, 1)).flatten()
    vectors = detector.extract_feature_vectors(
        data=normalized,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        pooling=pooling,
    )

    if output_path:
        detector.save_feature_vectors(vectors, output_path)

    return vectors
