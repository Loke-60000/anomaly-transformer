import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from pathlib import Path


class AnomalyVisualizer:
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

        sns.set_palette("husl")

    def plot_time_series_with_anomalies(
        self,
        data: np.ndarray,
        anomalies: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        title: str = "Time Series with Detected Anomalies",
        figsize: tuple = (15, 8),
        save_path: Optional[str] = None,
    ):
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        ax1 = axes[0]
        time_indices = np.arange(len(data))

        ax1.plot(time_indices, data, "b-", alpha=0.7, linewidth=1, label="Normal")

        if np.any(anomalies):
            anomaly_indices = time_indices[anomalies]
            anomaly_values = data[anomalies]
            ax1.scatter(
                anomaly_indices,
                anomaly_values,
                c="red",
                s=50,
                alpha=0.6,
                label="Anomaly",
                zorder=5,
            )

            in_anomaly = False
            start = 0
            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly and not in_anomaly:
                    start = i
                    in_anomaly = True
                elif not is_anomaly and in_anomaly:
                    ax1.axvspan(start, i - 1, alpha=0.2, color="red")
                    in_anomaly = False

            if in_anomaly:
                ax1.axvspan(start, len(anomalies) - 1, alpha=0.2, color="red")

        ax1.set_ylabel("Value", fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(
            time_indices[: len(scores)],
            scores,
            "g-",
            alpha=0.7,
            linewidth=1,
            label="Anomaly Score",
        )
        ax2.axhline(
            y=threshold,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold:.4f})",
        )
        ax2.fill_between(
            time_indices[: len(scores)],
            scores,
            threshold,
            where=(scores > threshold),
            alpha=0.3,
            color="red",
            label="Anomalous",
        )

        ax2.set_xlabel("Time Index", fontsize=12)
        ax2.set_ylabel("Anomaly Score", fontsize=12)
        ax2.set_title("Anomaly Scores", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_anomaly_distribution(
        self,
        scores: np.ndarray,
        threshold: float,
        title: str = "Anomaly Score Distribution",
        figsize: tuple = (12, 5),
        save_path: Optional[str] = None,
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax1 = axes[0]
        ax1.hist(scores, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax1.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold:.4f})",
        )
        ax1.set_xlabel("Anomaly Score", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Score Distribution", fontsize=13, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        normal_scores = scores[scores <= threshold]
        anomaly_scores = scores[scores > threshold]

        box_data = [normal_scores, anomaly_scores]
        box_labels = ["Normal", "Anomaly"]

        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        if len(box_data) > 1 and len(box_data[1]) > 0:
            bp["boxes"][1].set_facecolor("lightcoral")

        ax2.axhline(threshold, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax2.set_ylabel("Anomaly Score", fontsize=12)
        ax2.set_title("Score Comparison", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_training_history(
        self,
        history: Dict,
        title: str = "Training History",
        figsize: tuple = (12, 5),
        save_path: Optional[str] = None,
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        epochs = range(1, len(history["train_loss"]) + 1)

        ax1 = axes[0]
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(epochs, history["learning_rates"], "g-", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Learning Rate", fontsize=12)
        ax2.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_reconstruction(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        start_idx: int = 0,
        length: int = 100,
        title: str = "Original vs Reconstructed",
        figsize: tuple = (15, 5),
        save_path: Optional[str] = None,
    ):
        end_idx = min(start_idx + length, len(original))
        indices = range(start_idx, end_idx)

        plt.figure(figsize=figsize)
        plt.plot(
            indices,
            original[start_idx:end_idx],
            "b-",
            label="Original",
            linewidth=2,
            alpha=0.7,
        )
        plt.plot(
            indices,
            reconstructed[start_idx:end_idx],
            "r--",
            label="Reconstructed",
            linewidth=2,
            alpha=0.7,
        )

        error = np.abs(original[start_idx:end_idx] - reconstructed[start_idx:end_idx])
        plt.fill_between(
            indices,
            original[start_idx:end_idx],
            reconstructed[start_idx:end_idx],
            alpha=0.2,
            color="gray",
            label="Reconstruction Error",
        )

        plt.xlabel("Time Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(
            f"{title}\nMean Absolute Error: {error.mean():.4f}",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def create_summary_report(
        self,
        data: np.ndarray,
        results: Dict,
        history: Optional[Dict] = None,
        output_dir: str = "./plots",
    ):
        """Create a complete visual summary report with all charts."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("Generating visualization report...")

        point_scores = (
            np.array(results["point_scores"])
            if isinstance(results["point_scores"], list)
            else results["point_scores"]
        )
        point_anomalies = (
            np.array(results["point_anomalies"])
            if isinstance(results["point_anomalies"], list)
            else results["point_anomalies"]
        )

        self.plot_time_series_with_anomalies(
            data=data,
            anomalies=point_anomalies,
            scores=point_scores,
            threshold=results["threshold"],
            save_path=str(output_path / "anomalies_timeseries.png"),
        )

        self.plot_anomaly_distribution(
            scores=point_scores,
            threshold=results["threshold"],
            save_path=str(output_path / "score_distribution.png"),
        )

        if history:
            self.plot_training_history(
                history=history, save_path=str(output_path / "training_history.png")
            )

        print(f"\nVisualization report saved to {output_dir}/")
