#!/usr/bin/env python3
"""
Lightweight DSA Transformer Training Script for CPU
Optimized for running on a laptop CPU with smaller model size and batch size.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.improved_transformer import (
    AnomalyScoreCalculator,
    ImprovedTransformerAutoencoder,
)


class NASADataLoader:
    """Data loader for NASA SMAP/MSL dataset."""

    def __init__(self, data_path: str, window_size: int = 50, stride: int = 1):
        self.data_path = Path(data_path)
        self.window_size = window_size
        self.stride = stride

        # Load processed NASA data
        self.data = self._load_nasa_data()
        self.train_data, self.test_data, self.test_labels = self._prepare_data()

    def _load_nasa_data(self):
        """Load the processed NASA dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"NASA data not found at {self.data_path}")

        data = np.load(self.data_path)
        return {
            "train_data": data["train_sequences"],
            "test_data": data["test_sequences"],
            "test_labels": data["test_labels"],
            "feature_names": data.get(
                "feature_names", [f"feature_{i}" for i in range(25)]
            ),
        }

    def _prepare_data(self):
        """Prepare windowed sequences for training."""
        print("Preparing training data...")
        # Data is already windowed in the npz file
        train_sequences = self.data["train_data"]

        print("Preparing test data...")
        test_sequences = self.data["test_data"]
        test_labels = self.data["test_labels"]

        return train_sequences, test_sequences, test_labels

    def _create_windows(self, data):
        """Create sliding window sequences."""
        windows = []
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i : i + self.window_size]
            windows.append(window)
        return np.array(windows)

    def _create_window_labels(self, labels):
        """Create labels for windows (1 if any point in window is anomalous)."""
        window_labels = []
        for i in range(0, len(labels) - self.window_size + 1, self.stride):
            window_label = labels[i : i + self.window_size]
            # Window is anomalous if any point is anomalous
            window_labels.append(int(np.any(window_label)))
        return np.array(window_labels)

    def get_dataloaders(self, batch_size=32, train_split=0.8):
        """Create PyTorch DataLoaders."""
        # Split training data into train/validation
        n_train = int(len(self.train_data) * train_split)

        train_tensor = torch.FloatTensor(self.train_data[:n_train])
        val_tensor = torch.FloatTensor(self.train_data[n_train:])
        test_tensor = torch.FloatTensor(self.test_data)
        test_labels_tensor = torch.LongTensor(self.test_labels)

        # Create datasets
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


class EnhancedTrainer:
    """Trainer for the enhanced transformer model."""

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "epochs": []}
        self.best_val_loss = float("inf")
        self.anomaly_calculator = AnomalyScoreCalculator()

    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            reconstructed, attention_weights, latent_mean, latent_logvar = self.model(
                data
            )

            # Calculate losses
            recon_loss = criterion(reconstructed, data)
            kl_loss = self.model.kl_divergence_loss(latent_mean, latent_logvar)

            # Total loss
            loss = recon_loss + 0.01 * kl_loss  # Small weight for KL term

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:  # Print more frequently for small batches
                print(
                    f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )

        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(self.device)

                reconstructed, _, latent_mean, latent_logvar = self.model(data)

                # Calculate losses
                recon_loss = criterion(reconstructed, data)
                kl_loss = self.model.kl_divergence_loss(latent_mean, latent_logvar)
                loss = recon_loss + 0.01 * kl_loss

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=15):
        """Train the model."""
        print(f"Training enhanced transformer on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience // 3, factor=0.5
        )
        criterion = nn.MSELoss()

        no_improve = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)

            # Validation
            val_loss = self.validate_epoch(val_loader, criterion)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["epochs"].append(epoch + 1)

            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                no_improve = 0
                # Save best model
                self.save_checkpoint("best_dsa_cpu_model.pt", epoch, val_loss)
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

        print(f"\nTraining completed. Best validation loss: {self.best_val_loss:.6f}")

        return self.history

    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint_dir = PROJECT_ROOT / "assets" / "models" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "model_config": {
                "input_dim": self.model.input_dim,
                "d_model": self.model.d_model,
                "nhead": self.model.nhead,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
            },
        }

        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"Checkpoint saved: {filename}")

    def evaluate_anomaly_detection(self, test_loader):
        """Evaluate anomaly detection performance."""
        self.model.eval()
        all_scores = []
        all_labels = []

        print("Calculating anomaly scores...")

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)

                # Get model outputs
                reconstructed, attention_weights, latent_mean, latent_logvar = (
                    self.model(data)
                )

                # Calculate multiple anomaly scores
                scores = self.anomaly_calculator.calculate_comprehensive_scores(
                    data.cpu(),
                    reconstructed.cpu(),
                    attention_weights,
                    latent_mean.cpu(),
                    latent_logvar.cpu(),
                )

                all_scores.extend(scores["combined"].tolist())
                all_labels.extend(labels.tolist())

        # Convert to numpy arrays
        scores = np.array(all_scores)
        labels = np.array(all_labels)

        # Calculate metrics
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        auc = roc_auc_score(labels, scores)

        # Find optimal threshold
        thresholds = np.percentile(scores, [90, 95, 97, 99])
        best_f1 = 0
        best_threshold = None

        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average="binary"
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Final predictions with best threshold
        final_predictions = (scores > best_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, final_predictions, average="binary"
        )

        results = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "best_threshold": best_threshold,
            "confusion_matrix": confusion_matrix(labels, final_predictions).tolist(),
        }

        return results, scores, labels

    def plot_training_history(self):
        """Plot training history."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(
            self.history["epochs"],
            self.history["train_loss"],
            "b-",
            label="Training Loss",
        )
        plt.plot(
            self.history["epochs"],
            self.history["val_loss"],
            "r-",
            label="Validation Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            self.history["epochs"],
            self.history["train_loss"],
            "b-",
            label="Training Loss",
        )
        plt.plot(
            self.history["epochs"],
            self.history["val_loss"],
            "r-",
            label="Validation Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Training History (Log Scale)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save plot
        plots_dir = PROJECT_ROOT / "assets" / "outputs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            plots_dir / "dsa_cpu_training_history.png", dpi=300, bbox_inches="tight"
        )
        # plt.show()


def main():
    """Main training function."""
    print("Lightweight DSA Transformer Training for CPU")
    print("=" * 60)

    # Configuration
    config = {
        "data_path": PROJECT_ROOT
        / "assets"
        / "data"
        / "nasa"
        / "nasa_processed_data.npz",
        "window_size": 50,
        "batch_size": 16,  # Reduced for CPU
        "epochs": 5,  # Reduced for quick testing
        "learning_rate": 1e-3,
        "patience": 3,
        # Model configuration
        "input_dim": 25,
        "d_model": 64,  # Reduced model size
        "nhead": 4,  # Reduced heads
        "num_layers": 2,  # Reduced layers
        "dropout": 0.1,
        # DSA Configuration
        "use_dsa": True,
        "dsa_top_k": 8,  # Reduced top-k
        "dsa_d_index": 16,  # Reduced index dimension
        "dsa_num_heads": 2,  # Reduced index heads
    }

    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Check if data exists
    if not config["data_path"].exists():
        print(f"ERROR: NASA data not found at {config['data_path']}")
        print("Please run the NASA data preparation script first:")
        print("  python tools/scripts/data_processing/prepare_nasa_data.py")
        return

    # Load data
    print("Loading NASA SMAP/MSL dataset...")
    data_loader = NASADataLoader(config["data_path"], window_size=config["window_size"])

    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        batch_size=config["batch_size"]
    )

    print(f"Data loaded:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating enhanced transformer model with DSA...")
    model = ImprovedTransformerAutoencoder(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_dsa=config["use_dsa"],
        dsa_top_k=config["dsa_top_k"],
        dsa_d_index=config["dsa_d_index"],
        dsa_num_heads=config["dsa_num_heads"],
    )

    # Create trainer
    trainer = EnhancedTrainer(model)

    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=config["epochs"],
        lr=config["learning_rate"],
        patience=config["patience"],
    )

    # Plot training history
    trainer.plot_training_history()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results, scores, labels = trainer.evaluate_anomaly_detection(test_loader)

    print("\nAnomaly Detection Results:")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"  Best Threshold: {results['best_threshold']:.4f}")

    # Calculate False Positives and False Negatives
    tn, fp, fn, tp = np.array(results["confusion_matrix"]).ravel()
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    print(f"  True Negatives: {tn}")

    # Save results
    results_dir = PROJECT_ROOT / "assets" / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "dsa_cpu_results.json", "w") as f:
        json.dump(
            {
                "config": {k: str(v) for k, v in config.items()},
                "training_history": history,
                "evaluation_results": results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_dir / 'dsa_cpu_results.json'}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
