"""OOP pipeline orchestration for transformer-based anomaly detection."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import numpy as np

from .config.config import PipelineConfig, ConfigurationManager, PresetConfigurations
from .models.model import create_model, BaseAutoencoder
from .data.data_loader import AnomalyDataLoader
from .training.train import (
    AnomalyDetectionTrainer,
    TrainingConfig as TrainerConfig,
    CheckpointManager,
)


class AnomalyDetectionResults:
    """Encapsulates anomaly detection results."""

    def __init__(
        self,
        point_scores: np.ndarray,
        point_anomalies: np.ndarray,
        window_errors: np.ndarray,
        window_anomalies: np.ndarray,
        threshold: float,
        anomaly_segments: list,
    ):
        """
        Initialize results object.

        Args:
            point_scores: Anomaly scores for each point
            point_anomalies: Binary mask of point-level anomalies
            window_errors: Reconstruction errors for each window
            window_anomalies: Binary mask of window-level anomalies
            threshold: Threshold used for detection
            anomaly_segments: List of anomaly segments
        """
        self.point_scores = point_scores
        self.point_anomalies = point_anomalies
        self.window_errors = window_errors
        self.window_anomalies = window_anomalies
        self.threshold = threshold
        self.anomaly_segments = anomaly_segments

        self.n_anomalies = int(np.sum(point_anomalies))
        self.anomaly_rate = float(np.mean(point_anomalies))

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "point_scores": self.point_scores.tolist(),
            "point_anomalies": self.point_anomalies.tolist(),
            "window_errors": self.window_errors.tolist(),
            "window_anomalies": self.window_anomalies.tolist(),
            "threshold": float(self.threshold),
            "n_anomalies": self.n_anomalies,
            "anomaly_rate": self.anomaly_rate,
            "anomaly_segments": self.anomaly_segments,
        }

    def save(self, filepath: Path) -> None:
        """Save results to JSON file."""
        if isinstance(filepath, str):
            filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"✓ Results saved to {filepath}")

    def get_top_segments(self, n: int = 5) -> list:
        """Get top N anomaly segments by length."""
        sorted_segments = sorted(
            self.anomaly_segments, key=lambda x: x["length"], reverse=True
        )
        return sorted_segments[:n]

    def print_summary(self) -> None:
        """Print summary of detection results."""
        print("=" * 60)
        print("ANOMALY DETECTION RESULTS")
        print("=" * 60)
        print(f"Total anomalous points: {self.n_anomalies}")
        print(f"Anomaly rate: {self.anomaly_rate * 100:.2f}%")
        print(f"Threshold: {self.threshold:.6f}")
        print(f"Number of segments: {len(self.anomaly_segments)}")

        if self.anomaly_segments:
            print("\nTop 5 anomaly segments:")
            for i, segment in enumerate(self.get_top_segments(5), 1):
                start, end, length = segment["start"], segment["end"], segment["length"]
                avg_score = self.point_scores[start : end + 1].mean()
                print(
                    f"  {i}. Indices {start}-{end} (length: {length}, avg_score: {avg_score:.4f})"
                )

        print("=" * 60)


class AnomalyDetectionPipeline:
    """
    Main pipeline class that orchestrates the entire anomaly detection workflow.
    Follows OOP principles with clear separation of concerns.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config_manager = ConfigurationManager(config)

        self.data_loader: Optional[AnomalyDataLoader] = None
        self.model: Optional[BaseAutoencoder] = None
        self.trainer: Optional[AnomalyDetectionTrainer] = None
        self.results: Optional[AnomalyDetectionResults] = None

        if not self.config_manager.validate():
            raise ValueError("Invalid configuration")

    def setup_data(self) -> None:
        """Setup data loader."""
        print("\n" + "=" * 70)
        print("SETTING UP DATA")
        print("=" * 70)

        self.data_loader = AnomalyDataLoader(
            json_path=self.config.data.json_path,
            window_size=self.config.data.window_size,
            stride=self.config.data.stride,
            batch_size=self.config.data.batch_size,
            train_split=self.config.data.train_split,
            normalize=self.config.data.normalize,
            data_source=self.config.data.data_source,
            unit_id=self.config.data.unit_id,
            combine_method=self.config.data.combine_method,
        )

        print("✓ Data loader initialized")
        print(f"  Data source: {self.config.data.data_source}")
        if self.config.data.data_source in ["nodes", "index"]:
            print(f"  Unit ID: {self.config.data.unit_id}")
            if self.config.data.data_source == "index":
                print(f"  Combine method: {self.config.data.combine_method}")

    def setup_model(self) -> None:
        """Setup model."""
        print("\n" + "=" * 70)
        print("SETTING UP MODEL")
        print("=" * 70)

        model_kwargs = self.config.model.to_model_kwargs()
        self.model = create_model(self.config.model.model_type, **model_kwargs)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model created: {self.config.model.model_type}")
        print(f"  Parameters: {n_params:,}")

    def setup_trainer(self) -> None:
        print("\n" + "=" * 70)
        print("SETTING UP TRAINER")
        print("=" * 70)

        trainer_config = TrainerConfig(
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            epochs=self.config.training.epochs,
            early_stopping_patience=self.config.training.early_stopping_patience,
            grad_clip_norm=self.config.training.grad_clip_norm,
            checkpoint_dir=str(self.config.training.checkpoint_dir),
        )

        self.trainer = AnomalyDetectionTrainer(
            model=self.model,
            config=trainer_config,
            device=str(self.config.training.get_device()),
        )

        print(f"✓ Trainer initialized")
        print(f"  Device: {self.config.training.get_device()}")

    def train(self) -> Dict[str, Any]:
        """
        Execute training phase.

        Returns:
            Training history
        """
        if self.data_loader is None:
            self.setup_data()
        if self.model is None:
            self.setup_model()
        if self.trainer is None:
            self.setup_trainer()

        print("\n" + "=" * 70)
        print("TRAINING PHASE")
        print("=" * 70)

        # Load and process data
        train_loader, val_loader, raw_data = self.data_loader.load_and_process()

        print("\nData statistics:")
        for key, value in self.data_loader.get_statistics().items():
            print(f"  {key}: {value}")

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Train
        history = self.trainer.fit(train_loader, val_loader)

        print(f"\n✓ Training complete")
        print(f"  Best validation loss: {self.trainer.best_val_loss:.6f}")

        return history

    def detect_anomalies(
        self, data: Optional[np.ndarray] = None
    ) -> AnomalyDetectionResults:
        """
        Execute anomaly detection phase.

        Args:
            data: Optional data to detect on. If None, uses training data.

        Returns:
            Detection results
        """
        print("\n" + "=" * 70)
        print("ANOMALY DETECTION PHASE")
        print("=" * 70)

        if self.data_loader is None:
            self.setup_data()
            _, _, data = self.data_loader.load_and_process()

        if self.model is None:
            self.load_model()

        # Setup detector
        from .inference.inference import AnomalyDetector

        detector = AnomalyDetector(
            model=self.model,
            data_loader=self.data_loader,
            threshold_percentile=self.config.inference.threshold_percentile,
            device=str(self.config.inference.get_device()),
        )

        # Fit threshold on training data
        train_size = int(
            len(self.data_loader.processed_data) * self.config.data.train_split
        )
        train_data = self.data_loader.processed_data[:train_size]
        detector.fit_threshold(
            train_data, self.config.inference.window_size, self.config.inference.stride
        )

        # Detect
        results_dict = detector.detect_anomalies(
            data if data is not None else self.data_loader.raw_data,
            self.config.inference.window_size,
            self.config.inference.stride,
        )

        # Create results object
        self.results = AnomalyDetectionResults(
            point_scores=results_dict["point_scores"],
            point_anomalies=results_dict["point_anomalies"],
            window_errors=results_dict["window_errors"],
            window_anomalies=results_dict["window_anomalies"],
            threshold=results_dict["threshold"],
            anomaly_segments=results_dict["anomaly_segments"],
        )

        print(f"\n✓ Anomaly detection complete")
        self.results.print_summary()

        return self.results

    def visualize(self) -> None:
        """Generate visualizations."""
        print("\n" + "=" * 70)
        print("VISUALIZATION PHASE")
        print("=" * 70)

        if self.results is None:
            raise ValueError("No results to visualize. Run detect_anomalies() first.")

        from .utils.visualize import AnomalyVisualizer

        visualizer = AnomalyVisualizer(style=self.config.visualization.style)

        # Get data
        data = self.data_loader.raw_data

        # Load history if available
        history_path = self.config.training.checkpoint_dir / "training_history.json"
        history = None
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)

        # Create visualizations
        visualizer.create_summary_report(
            data=data,
            results=self.results.to_dict(),
            history=history,
            output_dir=str(self.config.visualization.output_dir),
        )

        print(f"\n✓ Visualizations saved to {self.config.visualization.output_dir}")

    def run_full_pipeline(self) -> AnomalyDetectionResults:
        """
        Execute the complete pipeline: train, detect, visualize.

        Returns:
            Detection results
        """
        print("\n" + "=" * 70)
        print("RUNNING FULL ANOMALY DETECTION PIPELINE")
        print("=" * 70)

        self.config_manager.print_summary()

        # Execute pipeline
        self.train()
        results = self.detect_anomalies()
        self.visualize()

        # Save results
        if self.config.inference.save_results:
            results_path = self.config.inference.output_dir / "anomaly_results.json"
            results.save(str(results_path))

        # Save configuration
        config_path = self.config.inference.output_dir / "pipeline_config.json"
        self.config_manager.save(str(config_path))

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(
            f"✓ Model checkpoint: {self.config.training.checkpoint_dir / 'best_model.pt'}"
        )
        print(f"✓ Results: {results_path}")
        print(f"✓ Visualizations: {self.config.visualization.output_dir}")
        print(f"✓ Configuration: {config_path}")

        return results

    def save_model(self, filepath: str) -> None:
        """
        Save the current model to a checkpoint file.

        Args:
            filepath: Path to save the checkpoint
        """
        if self.model is None:
            raise ValueError("No model to save")

        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "epoch": self.config.training.epochs if self.trainer else 0,
            "val_loss": (
                self.trainer.best_val_loss
                if self.trainer and hasattr(self.trainer, "best_val_loss")
                else 0.0
            ),
        }

        torch.save(checkpoint, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.inference.checkpoint_path
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create model if not exists
        if self.model is None:
            self.setup_model()

        # Load checkpoint
        device = self.config.inference.get_device()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Training epoch: {checkpoint['epoch']}")
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")

    def save_state(self, filepath: str) -> None:
        """Save pipeline state."""
        state = {
            "config": self.config.to_dict(),
            "has_model": self.model is not None,
            "has_data": self.data_loader is not None,
            "has_results": self.results is not None,
        }

        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        print(f"✓ Pipeline state saved to {filepath}")


class PipelineFactory:
    """Factory class for creating pipeline instances."""

    @staticmethod
    def create_from_config(config: PipelineConfig) -> AnomalyDetectionPipeline:
        """
        Create pipeline from configuration.

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline instance
        """
        return AnomalyDetectionPipeline(config)

    @staticmethod
    def create_from_preset(
        preset_name: str, json_path: str
    ) -> AnomalyDetectionPipeline:
        """
        Create pipeline from preset configuration.

        Args:
            preset_name: Name of preset ('quick_test', 'balanced', 'high_accuracy', 'fast_training', 'production')
            json_path: Path to data file

        Returns:
            Pipeline instance
        """
        preset_map = {
            "quick_test": PresetConfigurations.quick_test,
            "balanced": PresetConfigurations.balanced,
            "high_accuracy": PresetConfigurations.high_accuracy,
            "fast_training": PresetConfigurations.fast_training,
            "production": PresetConfigurations.production,
        }

        if preset_name not in preset_map:
            raise ValueError(
                f"Unknown preset: {preset_name}. Choose from {list(preset_map.keys())}"
            )

        config = preset_map[preset_name](json_path)
        return AnomalyDetectionPipeline(config)

    @staticmethod
    def create_from_json(config_path: str) -> AnomalyDetectionPipeline:
        """
        Create pipeline from JSON configuration file.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            Pipeline instance
        """
        config_manager = ConfigurationManager.load(config_path)
        return AnomalyDetectionPipeline(config_manager.config)


# Convenience function for quick pipeline execution
def run_anomaly_detection(
    json_path: str,
    preset: str = "balanced",
    train: bool = True,
    detect: bool = True,
    visualize: bool = True,
) -> AnomalyDetectionResults:
    """
    Convenience function to run anomaly detection with minimal setup.

    Args:
        json_path: Path to data file
        preset: Configuration preset name
        train: Whether to train model
        detect: Whether to detect anomalies
        visualize: Whether to generate visualizations

    Returns:
        Detection results
    """
    pipeline = PipelineFactory.create_from_preset(preset, json_path)

    if train:
        pipeline.train()
    else:
        pipeline.load_model()

    results = None
    if detect:
        results = pipeline.detect_anomalies()

    if visualize and results is not None:
        pipeline.visualize()

    return results
