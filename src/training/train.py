"""
Training module for transformer-based anomaly detection.
Provides trainer class and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, Tuple

from ..models.model import create_model, BaseAutoencoder
from ..data.data_loader import AnomalyDataLoader


class TrainingConfig:
    """Configuration for training."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        grad_clip_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)


class CheckpointManager:
    """Manages model checkpoints."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir

    def save(
        self,
        filename: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int,
        val_loss: float,
        history: Dict,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "history": history,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load(
        self,
        filename: str,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        device: str = "cpu",
    ) -> Tuple[int, float, Dict]:
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["epoch"], checkpoint["val_loss"], checkpoint["history"]


class TrainingHistory:
    """Tracks training metrics."""

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.learning_rates = []

    def update(self, train_loss: float, val_loss: float, lr: float):
        """Update history with new metrics."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.learning_rates.append(lr)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rates": self.learning_rates,
        }

    def save(self, filepath: str):
        """Save history to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class AnomalyDetectionTrainer:
    """Main trainer for anomaly detection models."""

    def __init__(
        self, model: BaseAutoencoder, config: TrainingConfig, device: str = None
    ):
        """
        Args:
            model: Autoencoder model
            config: Training configuration
            device: Device to train on (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config

        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.MSELoss()

        # Setup management
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.history = TrainingHistory()

        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            # Support both Dataset returning a Tensor and TensorDataset returning (Tensor, ...)
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(batch)
            loss = self.criterion(reconstructed, batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)

                total_loss += loss.item()
                n_batches += 1

        # Handle case where validation set is empty
        if n_batches == 0:
            return 0.0

        return total_loss / n_batches

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history dictionary
        """
        patience_counter = 0
        has_validation = len(val_loader) > 0
        val_loss = float("inf")  # Initialize val_loss

        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        if not has_validation:
            print(
                "Warning: No validation data available. Using training loss for monitoring."
            )

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate if validation set exists, otherwise use train loss
            if has_validation:
                val_loss = self.validate(val_loader)
            else:
                val_loss = train_loss  # Use training loss as proxy

            # Update learning rate
            self.scheduler.step(val_loss)

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.update(train_loss, val_loss, current_lr)

            print(
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt", epoch, val_loss)
                print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    f"checkpoint_epoch_{epoch + 1}.pt", epoch, val_loss
                )

        # Save final model and history
        self._save_checkpoint("final_model.pt", self.config.epochs, val_loss)
        self.history.save(self.config.checkpoint_dir / "training_history.json")

        return self.history.to_dict()

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save checkpoint."""
        self.checkpoint_manager.save(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            val_loss=val_loss,
            history=self.history.to_dict(),
        )


def train_anomaly_detector(
    data_path: str,
    model_type: str = "lightweight",
    window_size: int = 50,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = None,
) -> Tuple[BaseAutoencoder, AnomalyDataLoader, Dict]:
    """
    Main training function with simplified interface.

    Args:
        data_path: Path to JSON data file
        model_type: 'lightweight' or 'full'
        window_size: Sliding window size
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Tuple of (trained_model, data_loader, training_history)
    """
    # Load data
    print("Loading and preprocessing data...")
    data_loader = AnomalyDataLoader(
        json_path=data_path,
        window_size=window_size,
        batch_size=batch_size,
        train_split=0.8,
        normalize=True,
    )

    train_loader, val_loader, raw_data = data_loader.load_and_process()

    print("\nData statistics:")
    for key, value in data_loader.get_statistics().items():
        print(f"  {key}: {value}")

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\nCreating {model_type} model...")
    model_configs = {
        "lightweight": {
            "input_dim": 1,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
        },
        "full": {
            "input_dim": 1,
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
        },
    }

    model = create_model(model_type, **model_configs[model_type])

    # Create training configuration
    config = TrainingConfig(
        learning_rate=learning_rate, epochs=epochs, early_stopping_patience=15
    )

    # Create trainer and train
    trainer = AnomalyDetectionTrainer(model, config, device)

    print("\nStarting training...")
    history = trainer.fit(train_loader, val_loader)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")

    return model, data_loader, history


if __name__ == "__main__":
    model, data_loader, history = train_anomaly_detector(
        data_path="./data-output.json",
        model_type="lightweight",
        window_size=50,
        batch_size=16,
        epochs=100,
        learning_rate=1e-3,
    )
