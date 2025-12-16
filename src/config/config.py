"""
Configuration classes for the anomaly detection system.
All configuration is defined using OOP principles with PyTorch integration.
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    json_path: str
    data_source: str = "json"  # "json" for old format, "nodes" for timeseries-data
    unit_id: str = "73"  # Unit ID for node data (e.g., "73" for temperature)
    combine_method: str = "concatenate"  # "concatenate" or "average" for multiple nodes
    window_size: int = 50
    stride: int = 1
    batch_size: int = 32
    train_split: float = 0.8
    normalize: bool = True
    missing_values: List[float] = field(default_factory=lambda: [0, -1])
    num_workers: int = 0
    pin_memory: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.train_split < 1:
            raise ValueError(
                f"train_split must be between 0 and 1, got {self.train_split}"
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.data_source not in ["json", "nodes", "index"]:
            raise ValueError(
                f"data_source must be 'json', 'nodes', or 'index', got {self.data_source}"
            )

        if not Path(self.json_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.json_path}")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_type: str = "lightweight"  # 'lightweight' or 'full'
    input_dim: int = 1
    d_model: int = 32
    nhead: int = 2
    num_layers: int = 2
    num_encoder_layers: Optional[int] = None
    num_decoder_layers: Optional[int] = None
    dim_feedforward: int = 128
    dropout: float = 0.1
    activation: str = "gelu"

    def __post_init__(self):
        """Set defaults based on model type."""
        if self.model_type == "lightweight":
            self.d_model = 32
            self.nhead = 2
            self.num_layers = 2
            self.dim_feedforward = 128
        elif self.model_type == "full":
            self.d_model = 64
            self.nhead = 4
            self.num_encoder_layers = 3
            self.num_decoder_layers = 3
            self.dim_feedforward = 256
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def validate(self) -> None:
        """Validate model configuration."""
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for model creation."""
        kwargs = {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }

        if self.model_type == "lightweight":
            kwargs["num_layers"] = self.num_layers
        else:
            kwargs["num_encoder_layers"] = self.num_encoder_layers
            kwargs["num_decoder_layers"] = self.num_decoder_layers
            kwargs["activation"] = self.activation

        return kwargs


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    save_frequency: int = 10
    device: Optional[str] = None
    use_amp: bool = False  # Automatic Mixed Precision
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    def __post_init__(self):
        """Initialize and validate configuration."""
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def validate(self) -> None:
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

    def get_device(self) -> torch.device:
        """Get PyTorch device for training."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class InferenceConfig:
    """Configuration for anomaly detection inference."""

    checkpoint_path: Path = field(
        default_factory=lambda: Path("./checkpoints/best_model.pt")
    )
    threshold_percentile: float = 95.0
    window_size: int = 50
    stride: int = 1
    device: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    save_results: bool = True

    def __post_init__(self):
        """Initialize and validate configuration."""
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def validate(self) -> None:
        """Validate inference configuration."""
        if not 0 < self.threshold_percentile <= 100:
            raise ValueError(
                f"threshold_percentile must be between 0 and 100, got {self.threshold_percentile}"
            )
        # Note: We don't check if checkpoint exists here because it may not exist yet during training

    def get_device(self) -> torch.device:
        """Get PyTorch device for inference."""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    output_dir: Path = field(default_factory=lambda: Path("./plots"))
    style: str = "seaborn-v0_8-darkgrid"
    figsize_timeseries: tuple = (15, 8)
    figsize_distribution: tuple = (12, 5)
    figsize_history: tuple = (12, 5)
    dpi: int = 300
    save_format: str = "png"
    show_plots: bool = False

    def __post_init__(self):
        """Initialize configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def validate(self) -> None:
        """Validate visualization configuration."""
        if self.dpi < 1:
            raise ValueError(f"dpi must be positive, got {self.dpi}")


class PipelineConfig:
    """Complete configuration for the entire pipeline."""

    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        training: TrainingConfig,
        inference: InferenceConfig,
        visualization: VisualizationConfig,
    ):
        """
        Initialize pipeline configuration.

        Args:
            data: Data configuration
            model: Model configuration
            training: Training configuration
            inference: Inference configuration
            visualization: Visualization configuration
        """
        self.data = data
        self.model = model
        self.training = training
        self.inference = inference
        self.visualization = visualization

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            inference=InferenceConfig(**config_dict.get("inference", {})),
            visualization=VisualizationConfig(**config_dict.get("visualization", {})),
        )

    @classmethod
    def default(cls, json_path: str) -> "PipelineConfig":
        """Create default configuration."""
        return cls(
            data=DataConfig(json_path=json_path),
            model=ModelConfig(model_type="lightweight"),
            training=TrainingConfig(),
            inference=InferenceConfig(),
            visualization=VisualizationConfig(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.data.__dict__.items()
            },
            "model": self.model.__dict__,
            "training": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.training.__dict__.items()
            },
            "inference": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.inference.__dict__.items()
            },
            "visualization": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.visualization.__dict__.items()
            },
        }

    def validate_all(self) -> bool:
        """Validate all configurations."""
        try:
            self.data.validate()
            self.model.validate()
            self.training.validate()
            self.inference.validate()
            self.visualization.validate()
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


class ConfigurationManager:
    """Manages configuration loading, saving, and validation."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize configuration manager.

        Args:
            config: Pipeline configuration object
        """
        self.config = config

    def save(self, filepath: Path) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save configuration
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        with open(filepath, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"✓ Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "ConfigurationManager":
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            ConfigurationManager instance
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        config = PipelineConfig.from_dict(config_dict)
        print(f"✓ Configuration loaded from {filepath}")

        return cls(config)

    def validate(self) -> bool:
        """Validate entire configuration."""
        return self.config.validate_all()

    def print_summary(self) -> None:
        """Print configuration summary."""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)

        print("\n[DATA]")
        print(f"  JSON Path: {self.config.data.json_path}")
        print(f"  Window Size: {self.config.data.window_size}")
        print(f"  Batch Size: {self.config.data.batch_size}")
        print(f"  Train Split: {self.config.data.train_split:.1%}")
        print(f"  Normalize: {self.config.data.normalize}")

        print("\n[MODEL]")
        print(f"  Type: {self.config.model.model_type}")
        print(f"  D-Model: {self.config.model.d_model}")
        print(f"  Attention Heads: {self.config.model.nhead}")
        print(
            f"  Layers: {self.config.model.num_layers if self.config.model.model_type == 'lightweight' else f'{self.config.model.num_encoder_layers}/{self.config.model.num_decoder_layers} (enc/dec)'}"
        )
        print(f"  Dropout: {self.config.model.dropout}")

        print("\n[TRAINING]")
        print(f"  Learning Rate: {self.config.training.learning_rate}")
        print(f"  Epochs: {self.config.training.epochs}")
        print(
            f"  Early Stopping: {self.config.training.early_stopping_patience} epochs"
        )
        print(f"  Device: {self.config.training.get_device()}")
        print(f"  Mixed Precision: {self.config.training.use_amp}")
        print(f"  Checkpoint Dir: {self.config.training.checkpoint_dir}")

        print("\n[INFERENCE]")
        print(f"  Threshold: {self.config.inference.threshold_percentile}th percentile")
        print(f"  Window Size: {self.config.inference.window_size}")
        print(f"  Device: {self.config.inference.get_device()}")
        print(f"  Output Dir: {self.config.inference.output_dir}")

        print("\n[VISUALIZATION]")
        print(f"  Output Dir: {self.config.visualization.output_dir}")
        print(f"  DPI: {self.config.visualization.dpi}")
        print(f"  Format: {self.config.visualization.save_format}")

        print("=" * 70)


class PresetConfigurations:
    """Factory class for predefined configuration presets."""

    @staticmethod
    def quick_test(json_path: str) -> PipelineConfig:
        """
        Fast configuration for testing and debugging.

        Args:
            json_path: Path to data file

        Returns:
            Quick test configuration
        """
        return PipelineConfig(
            data=DataConfig(json_path=json_path, window_size=20, batch_size=16),
            model=ModelConfig(model_type="lightweight", d_model=16, num_layers=1),
            training=TrainingConfig(epochs=10, early_stopping_patience=3),
            inference=InferenceConfig(threshold_percentile=95.0),
            visualization=VisualizationConfig(dpi=150),
        )

    @staticmethod
    def balanced(json_path: str) -> PipelineConfig:
        """
        Balanced configuration (recommended default).

        Args:
            json_path: Path to data file

        Returns:
            Balanced configuration
        """
        return PipelineConfig.default(json_path)

    @staticmethod
    def high_accuracy(json_path: str) -> PipelineConfig:
        """
        Configuration optimized for maximum accuracy.

        Args:
            json_path: Path to data file

        Returns:
            High accuracy configuration
        """
        return PipelineConfig(
            data=DataConfig(json_path=json_path, window_size=100, batch_size=32),
            model=ModelConfig(
                model_type="full",
                d_model=128,
                nhead=8,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=512,
                dropout=0.2,
            ),
            training=TrainingConfig(
                epochs=200, learning_rate=5e-4, early_stopping_patience=25, use_amp=True
            ),
            inference=InferenceConfig(threshold_percentile=99.0),
            visualization=VisualizationConfig(),
        )

    @staticmethod
    def fast_training(json_path: str) -> PipelineConfig:
        """
        Configuration optimized for training speed.

        Args:
            json_path: Path to data file

        Returns:
            Fast training configuration
        """
        return PipelineConfig(
            data=DataConfig(json_path=json_path, window_size=30, batch_size=64),
            model=ModelConfig(
                model_type="lightweight",
                d_model=16,
                nhead=2,
                num_layers=1,
                dropout=0.05,
            ),
            training=TrainingConfig(
                epochs=50, learning_rate=2e-3, early_stopping_patience=5
            ),
            inference=InferenceConfig(),
            visualization=VisualizationConfig(),
        )

    @staticmethod
    def production(json_path: str) -> PipelineConfig:
        """
        Configuration optimized for production deployment.

        Args:
            json_path: Path to data file

        Returns:
            Production configuration
        """
        return PipelineConfig(
            data=DataConfig(
                json_path=json_path,
                window_size=50,
                batch_size=32,
                num_workers=4,
                pin_memory=True,
            ),
            model=ModelConfig(model_type="full"),
            training=TrainingConfig(
                epochs=150, use_amp=True, checkpoint_dir=Path("./models/production")
            ),
            inference=InferenceConfig(
                checkpoint_path=Path("./models/production/best_model.pt"),
                threshold_percentile=97.0,
                output_dir=Path("./outputs/production"),
            ),
            visualization=VisualizationConfig(output_dir=Path("./reports/production")),
        )
