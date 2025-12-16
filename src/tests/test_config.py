"""
Unit tests for configuration classes.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path

import sys

sys.path.append("..")

from src.config.config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    VisualizationConfig,
    PipelineConfig,
    ConfigurationManager,
    PresetConfigurations,
)


class TestDataConfig:
    """Test DataConfig class."""

    def test_data_config_creation(self, tmp_path):
        """Test basic DataConfig creation."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = DataConfig(json_path=str(test_file))
        assert config.window_size == 50
        assert config.batch_size == 32
        assert config.train_split == 0.8
        assert config.normalize is True

    def test_data_config_validation_train_split(self, tmp_path):
        """Test train_split validation."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        # Invalid train_split
        config = DataConfig(json_path=str(test_file), train_split=1.5)
        with pytest.raises(ValueError, match="train_split must be between 0 and 1"):
            config.validate()

    def test_data_config_validation_window_size(self, tmp_path):
        """Test window_size validation."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = DataConfig(json_path=str(test_file), window_size=-1)
        with pytest.raises(ValueError, match="window_size must be positive"):
            config.validate()

    def test_data_config_validation_batch_size(self, tmp_path):
        """Test batch_size validation."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = DataConfig(json_path=str(test_file), batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_data_config_file_not_found(self):
        """Test file not found error."""
        config = DataConfig(json_path="nonexistent.json")
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            config.validate()


class TestModelConfig:
    """Test ModelConfig class."""

    def test_lightweight_model_config(self):
        """Test lightweight model configuration."""
        config = ModelConfig(model_type="lightweight")
        assert config.d_model == 32
        assert config.nhead == 2
        assert config.num_layers == 2
        assert config.dim_feedforward == 128

    def test_full_model_config(self):
        """Test full model configuration."""
        config = ModelConfig(model_type="full")
        assert config.d_model == 64
        assert config.nhead == 4
        assert config.num_encoder_layers == 3
        assert config.num_decoder_layers == 3
        assert config.dim_feedforward == 256

    def test_invalid_model_type(self):
        """Test invalid model type."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            ModelConfig(model_type="invalid")

    def test_model_config_validation_divisibility(self):
        """Test d_model divisibility by nhead."""
        config = ModelConfig(model_type="lightweight", d_model=33, nhead=2)
        with pytest.raises(ValueError, match="d_model.*must be divisible by nhead"):
            config.validate()

    def test_model_config_validation_d_model(self):
        """Test d_model positive validation."""
        config = ModelConfig(model_type="lightweight", d_model=-1)
        with pytest.raises(ValueError, match="d_model must be positive"):
            config.validate()

    def test_model_config_to_model_kwargs_lightweight(self):
        """Test conversion to model kwargs for lightweight."""
        config = ModelConfig(model_type="lightweight")
        kwargs = config.to_model_kwargs()

        assert "input_dim" in kwargs
        assert "d_model" in kwargs
        assert "num_layers" in kwargs
        assert "num_encoder_layers" not in kwargs

    def test_model_config_to_model_kwargs_full(self):
        """Test conversion to model kwargs for full model."""
        config = ModelConfig(model_type="full")
        kwargs = config.to_model_kwargs()

        assert "input_dim" in kwargs
        assert "d_model" in kwargs
        assert "num_encoder_layers" in kwargs
        assert "num_decoder_layers" in kwargs
        assert "activation" in kwargs


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_training_config_creation(self):
        """Test basic TrainingConfig creation."""
        config = TrainingConfig()
        assert config.learning_rate == 1e-3
        assert config.epochs == 100
        assert config.early_stopping_patience == 15
        assert config.use_amp is False

    def test_training_config_checkpoint_dir_creation(self, tmp_path):
        """Test checkpoint directory creation."""
        checkpoint_dir = tmp_path / "checkpoints"
        config = TrainingConfig(checkpoint_dir=checkpoint_dir)
        assert checkpoint_dir.exists()

    def test_training_config_validation_learning_rate(self):
        """Test learning rate validation."""
        config = TrainingConfig(learning_rate=-0.001)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()

    def test_training_config_validation_epochs(self):
        """Test epochs validation."""
        config = TrainingConfig(epochs=0)
        with pytest.raises(ValueError, match="epochs must be positive"):
            config.validate()

    def test_training_config_get_device(self):
        """Test device selection."""
        config = TrainingConfig()
        device = config.get_device()
        assert isinstance(device, torch.device)

    def test_training_config_get_device_explicit(self):
        """Test explicit device selection."""
        config = TrainingConfig(device="cpu")
        device = config.get_device()
        assert device.type == "cpu"


class TestInferenceConfig:
    """Test InferenceConfig class."""

    def test_inference_config_creation(self):
        """Test basic InferenceConfig creation."""
        config = InferenceConfig()
        assert config.threshold_percentile == 95.0
        assert config.window_size == 50
        assert config.save_results is True

    def test_inference_config_output_dir_creation(self, tmp_path):
        """Test output directory creation."""
        output_dir = tmp_path / "results"
        config = InferenceConfig(output_dir=output_dir)
        assert output_dir.exists()

    def test_inference_config_validation_threshold(self):
        """Test threshold percentile validation."""
        config = InferenceConfig(threshold_percentile=150)
        with pytest.raises(
            ValueError, match="threshold_percentile must be between 0 and 100"
        ):
            config.validate()

    def test_inference_config_get_device(self):
        """Test device selection."""
        config = InferenceConfig()
        device = config.get_device()
        assert isinstance(device, torch.device)


class TestVisualizationConfig:
    """Test VisualizationConfig class."""

    def test_visualization_config_creation(self):
        """Test basic VisualizationConfig creation."""
        config = VisualizationConfig()
        assert config.dpi == 300
        assert config.save_format == "png"
        assert config.show_plots is False

    def test_visualization_config_validation_dpi(self):
        """Test DPI validation."""
        config = VisualizationConfig(dpi=-100)
        with pytest.raises(ValueError, match="dpi must be positive"):
            config.validate()


class TestPipelineConfig:
    """Test PipelineConfig class."""

    def test_pipeline_config_creation(self, tmp_path):
        """Test basic PipelineConfig creation."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PipelineConfig.default(str(test_file))
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.inference, InferenceConfig)
        assert isinstance(config.visualization, VisualizationConfig)

    def test_pipeline_config_to_dict(self, tmp_path):
        """Test conversion to dictionary."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PipelineConfig.default(str(test_file))
        config_dict = config.to_dict()

        assert "data" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert "inference" in config_dict
        assert "visualization" in config_dict

    def test_pipeline_config_from_dict(self, tmp_path):
        """Test creation from dictionary."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config_dict = {
            "data": {"json_path": str(test_file), "window_size": 50},
            "model": {"model_type": "lightweight"},
            "training": {"epochs": 100},
            "inference": {"threshold_percentile": 95.0},
            "visualization": {"dpi": 300},
        }

        config = PipelineConfig.from_dict(config_dict)
        assert config.data.window_size == 50
        assert config.model.model_type == "lightweight"
        assert config.training.epochs == 100


class TestConfigurationManager:
    """Test ConfigurationManager class."""

    def test_configuration_manager_save_load(self, tmp_path):
        """Test saving and loading configuration."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        # Create and save configuration
        config = PipelineConfig.default(str(test_file))
        manager = ConfigurationManager(config)

        save_path = tmp_path / "config.json"
        manager.save(save_path)

        assert save_path.exists()

        # Load configuration
        loaded_manager = ConfigurationManager.load(save_path)
        assert loaded_manager.config.data.window_size == 50
        assert loaded_manager.config.model.model_type == "lightweight"

    def test_configuration_manager_load_nonexistent(self, tmp_path):
        """Test loading nonexistent configuration file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigurationManager.load(tmp_path / "nonexistent.json")

    def test_configuration_manager_print_summary(self, tmp_path, capsys):
        """Test printing configuration summary."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PipelineConfig.default(str(test_file))
        manager = ConfigurationManager(config)

        manager.print_summary()
        captured = capsys.readouterr()
        assert "CONFIGURATION SUMMARY" in captured.out
        assert "[DATA]" in captured.out
        assert "[MODEL]" in captured.out


class TestPresetConfigurations:
    """Test PresetConfigurations factory class."""

    def test_quick_test_preset(self, tmp_path):
        """Test quick_test preset."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PresetConfigurations.quick_test(str(test_file))
        assert config.data.window_size == 20
        assert config.model.d_model == 16
        assert config.training.epochs == 10

    def test_balanced_preset(self, tmp_path):
        """Test balanced preset."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PresetConfigurations.balanced(str(test_file))
        assert config.data.window_size == 50
        assert config.model.model_type == "lightweight"

    def test_high_accuracy_preset(self, tmp_path):
        """Test high_accuracy preset."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PresetConfigurations.high_accuracy(str(test_file))
        assert config.data.window_size == 100
        assert config.model.model_type == "full"
        assert config.model.d_model == 128
        assert config.training.epochs == 200

    def test_fast_training_preset(self, tmp_path):
        """Test fast_training preset."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PresetConfigurations.fast_training(str(test_file))
        assert config.data.batch_size == 64
        assert config.model.num_layers == 1
        assert config.training.epochs == 50

    def test_production_preset(self, tmp_path):
        """Test production preset."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": {"values": [[]]}}')

        config = PresetConfigurations.production(str(test_file))
        assert config.data.num_workers == 4
        assert config.training.use_amp is True
        assert config.training.epochs == 150
