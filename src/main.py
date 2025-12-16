"""
Main entry point for the transformer-based anomaly detection system.
Demonstrates full OOP usage of the pipeline.
"""

from .pipeline import AnomalyDetectionPipeline, PipelineFactory, run_anomaly_detection
from .config.config import (
    PipelineConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    VisualizationConfig,
    PresetConfigurations,
    ConfigurationManager,
)


class AnomalyDetectionApp:
    """Main application class for anomaly detection."""

    def __init__(self, data_path: str):
        """
        Initialize application.

        Args:
            data_path: Path to data file
        """
        self.data_path = data_path
        self.pipeline: AnomalyDetectionPipeline = None

    def run_with_preset(self, preset_name: str = "balanced") -> None:
        """
        Run anomaly detection with a preset configuration.

        Args:
            preset_name: Name of preset configuration
        """
        print(f"\n{'=' * 70}")
        print(f"RUNNING WITH PRESET: {preset_name.upper()}")
        print(f"{'=' * 70}\n")

        self.pipeline = PipelineFactory.create_from_preset(preset_name, self.data_path)
        results = self.pipeline.run_full_pipeline()

        return results

    def run_with_custom_config(self, config: PipelineConfig) -> None:
        """
        Run anomaly detection with custom configuration.

        Args:
            config: Custom pipeline configuration
        """
        print(f"\n{'=' * 70}")
        print("RUNNING WITH CUSTOM CONFIGURATION")
        print(f"{'=' * 70}\n")

        self.pipeline = AnomalyDetectionPipeline(config)
        results = self.pipeline.run_full_pipeline()

        return results

    def run_quick_test(self) -> None:
        """Run a quick test for debugging."""
        return self.run_with_preset("quick_test")

    def run_balanced(self) -> None:
        """Run with balanced configuration (recommended)."""
        return self.run_with_preset("balanced")

    def run_high_accuracy(self) -> None:
        """Run with high accuracy configuration."""
        return self.run_with_preset("high_accuracy")

    def run_production(self) -> None:
        """Run with production configuration."""
        return self.run_with_preset("production")


class ExampleUsages:
    """Examples demonstrating different ways to use the system."""

    @staticmethod
    def example_1_simple():
        """Example 1: Simplest usage with preset."""
        print("\n" + "=" * 70)
        print("EXAMPLE 1: Simple Usage with Preset")
        print("=" * 70)

        # Create and run pipeline with one line
        results = run_anomaly_detection(
            json_path="./data-output.json", preset="balanced"
        )

        print(f"\nDetected {results.n_anomalies} anomalies!")

    @staticmethod
    def example_2_preset_factory():
        """Example 2: Using factory with preset."""
        print("\n" + "=" * 70)
        print("EXAMPLE 2: Using Factory with Preset")
        print("=" * 70)

        # Create pipeline from preset
        pipeline = PipelineFactory.create_from_preset(
            preset_name="balanced", json_path="./data-output.json"
        )

        # Run full pipeline
        results = pipeline.run_full_pipeline()

        print(f"\nAnom aly rate: {results.anomaly_rate * 100:.2f}%")

    @staticmethod
    def example_3_custom_config():
        """Example 3: Custom configuration."""
        print("\n" + "=" * 70)
        print("EXAMPLE 3: Custom Configuration")
        print("=" * 70)

        # Create custom configuration
        config = PipelineConfig(
            data=DataConfig(
                json_path="./data-output.json", window_size=60, batch_size=24
            ),
            model=ModelConfig(
                model_type="lightweight", d_model=48, nhead=4, num_layers=3
            ),
            training=TrainingConfig(epochs=80, learning_rate=5e-4),
            inference=InferenceConfig(threshold_percentile=97.0),
            visualization=VisualizationConfig(dpi=200),
        )

        # Create and run pipeline
        pipeline = AnomalyDetectionPipeline(config)
        results = pipeline.run_full_pipeline()

        print(f"\nFound {len(results.anomaly_segments)} anomaly segments")

    @staticmethod
    def example_4_app_class():
        """Example 4: Using the App class."""
        print("\n" + "=" * 70)
        print("EXAMPLE 4: Using App Class")
        print("=" * 70)

        # Create app instance
        app = AnomalyDetectionApp(data_path="./data-output.json")

        # Run with different presets
        results = app.run_balanced()

        print(f"\nTop anomaly segment: {results.get_top_segments(1)[0]}")

    @staticmethod
    def example_5_step_by_step():
        """Example 5: Step-by-step execution."""
        print("\n" + "=" * 70)
        print("EXAMPLE 5: Step-by-Step Execution")
        print("=" * 70)

        # Create configuration
        config = PresetConfigurations.balanced("./data-output.json")

        # Create pipeline
        pipeline = AnomalyDetectionPipeline(config)

        # Execute steps individually
        print("\n[1] Training...")
        history = pipeline.train()
        print(f"    Final train loss: {history['train_loss'][-1]:.6f}")

        print("\n[2] Detecting anomalies...")
        results = pipeline.detect_anomalies()
        print(f"    Detected {results.n_anomalies} anomalies")

        print("\n[3] Visualizing...")
        pipeline.visualize()
        print("    Visualizations created")

        # Save results
        results.save("./custom_results.json")
        print("\n[4] Results saved!")

    @staticmethod
    def example_6_config_management():
        """Example 6: Configuration management."""
        print("\n" + "=" * 70)
        print("EXAMPLE 6: Configuration Management")
        print("=" * 70)

        # Create configuration
        config = PresetConfigurations.high_accuracy("./data-output.json")

        # Save configuration
        config_manager = ConfigurationManager(config)
        config_manager.save("./my_config.json")

        # Validate
        is_valid = config_manager.validate()
        print(f"\nConfiguration valid: {is_valid}")

        # Print summary
        config_manager.print_summary()

        # Load configuration later
        loaded_manager = ConfigurationManager.load("./my_config.json")

        # Create pipeline from loaded config
        pipeline = AnomalyDetectionPipeline(loaded_manager.config)
        print("\n✓ Pipeline created from saved configuration")

    @staticmethod
    def example_7_inference_only():
        """Example 7: Inference only (model already trained)."""
        print("\n" + "=" * 70)
        print("EXAMPLE 7: Inference Only")
        print("=" * 70)

        # Create configuration for inference
        config = PipelineConfig.default("./data-output.json")

        # Create pipeline
        pipeline = AnomalyDetectionPipeline(config)

        # Load pre-trained model
        pipeline.load_model("./checkpoints/best_model.pt")

        # Detect anomalies
        results = pipeline.detect_anomalies()

        # Visualize
        pipeline.visualize()

        print(f"\n✓ Inference complete: {results.n_anomalies} anomalies detected")


def main():
    """Main function demonstrating the recommended usage."""

    print("\n" + "=" * 70)
    print("TRANSFORMER-BASED ANOMALY DETECTION")
    print("Complete OOP Implementation")
    print("=" * 70)

    # Option 1: Simplest approach - one function call
    print("\n[RECOMMENDED] Running with preset configuration...")
    results = run_anomaly_detection(
        json_path="./data-output.json",
        preset="balanced",  # or 'quick_test', 'high_accuracy', 'fast_training', 'production'
        train=True,
        detect=True,
        visualize=True,
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    results.print_summary()

    print("\n✓ All done! Check the following directories:")
    print("  - ./checkpoints/  - Model checkpoints")
    print("  - ./results/      - Detection results")
    print("  - ./plots/        - Visualizations")


def run_all_examples():
    """Run all example usages."""
    examples = ExampleUsages()

    # Run each example
    try:
        examples.example_1_simple()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        examples.example_2_preset_factory()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        examples.example_3_custom_config()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        examples.example_4_app_class()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        examples.example_5_step_by_step()
    except Exception as e:
        print(f"Example 5 failed: {e}")

    try:
        examples.example_6_config_management()
    except Exception as e:
        print(f"Example 6 failed: {e}")

    try:
        examples.example_7_inference_only()
    except Exception as e:
        print(f"Example 7 failed: {e}")


if __name__ == "__main__":
    # Run main pipeline
    main()

    # Uncomment to run all examples:
    # run_all_examples()
