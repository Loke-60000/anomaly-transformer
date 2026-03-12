import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import json
import time
from typing import Dict, Tuple

try:
    from IPython.display import display, HTML
    from IPython import get_ipython

    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False

from ..models.model import create_model, BaseAutoencoder
from ..data.data_loader import AnomalyDataLoader


class TrainingConfig:
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
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)


class CheckpointManager:
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
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["epoch"], checkpoint["val_loss"], checkpoint["history"]


class TrainingHistory:
    def __init__(self):
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.learning_rates: list[float] = []
        self.epoch_times: list[float] = []

    def update(
        self, train_loss: float, val_loss: float, lr: float, elapsed: float = 0.0
    ):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.learning_rates.append(lr)
        self.epoch_times.append(elapsed)

    def to_dict(self) -> Dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        n = len(self.train_loss)
        if n == 0:
            return "TrainingHistory(empty)"
        return (
            f"TrainingHistory({n} epochs, "
            f"best_val={min(self.val_loss):.6f}, "
            f"final_train={self.train_loss[-1]:.6f})"
        )

    def summary_table(self, last_n: int = 0) -> str:
        rows = list(range(len(self.train_loss)))
        if last_n > 0:
            rows = rows[-last_n:]

        header = f"{'Epoch':>6} │ {'Train Loss':>12} │ {'Val Loss':>12} │ {'LR':>10} │ {'Time (s)':>9}"
        sep = "─" * len(header)
        lines = [sep, header, sep]
        best_val = min(self.val_loss)
        for i in rows:
            marker = " ★" if self.val_loss[i] == best_val else ""
            t = self.epoch_times[i] if i < len(self.epoch_times) else 0.0
            lines.append(
                f"{i + 1:>6} │ {self.train_loss[i]:>12.6f} │ {self.val_loss[i]:>12.6f} │ {self.learning_rates[i]:>10.2e} │ {t:>8.1f}s{marker}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def display_table(self, last_n: int = 0):
        if _HAS_IPYTHON:
            rows = list(range(len(self.train_loss)))
            if last_n > 0:
                rows = rows[-last_n:]
            best_val = min(self.val_loss)
            html = [
                "<table style='font-family:monospace; border-collapse:collapse'>",
                "<tr style='background:#f0f0f0'>",
                "<th style='padding:4px 10px'>Epoch</th>",
                "<th style='padding:4px 10px'>Train Loss</th>",
                "<th style='padding:4px 10px'>Val Loss</th>",
                "<th style='padding:4px 10px'>LR</th>",
                "<th style='padding:4px 10px'>Time</th></tr>",
            ]
            for i in rows:
                bg = "#e6ffe6" if self.val_loss[i] == best_val else ""
                t = self.epoch_times[i] if i < len(self.epoch_times) else 0.0
                html.append(
                    f"<tr style='background:{bg}'>"
                    f"<td style='padding:2px 10px; text-align:right'>{i+1}</td>"
                    f"<td style='padding:2px 10px; text-align:right'>{self.train_loss[i]:.6f}</td>"
                    f"<td style='padding:2px 10px; text-align:right'>{self.val_loss[i]:.6f}</td>"
                    f"<td style='padding:2px 10px; text-align:right'>{self.learning_rates[i]:.2e}</td>"
                    f"<td style='padding:2px 10px; text-align:right'>{t:.1f}s</td></tr>"
                )
            html.append("</table>")
            display(HTML("\n".join(html)))
        else:
            print(self.summary_table(last_n=last_n))

    def plot_learning_curves(self, figsize: tuple = (12, 4)):
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.train_loss) + 1)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax = axes[0]
        ax.plot(epochs, self.train_loss, label="Train Loss", linewidth=1.5)
        ax.plot(epochs, self.val_loss, label="Val Loss", linewidth=1.5)
        best_epoch = int(np.argmin(self.val_loss)) + 1
        best_val = min(self.val_loss)
        ax.axvline(
            best_epoch,
            color="grey",
            linestyle="--",
            alpha=0.5,
            label=f"Best epoch {best_epoch}",
        )
        ax.scatter([best_epoch], [best_val], marker="*", s=120, color="red", zorder=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1]
        ax.plot(epochs, self.learning_rates, color="tab:orange", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class AnomalyDetectionTrainer:
    def __init__(
        self,
        model: BaseAutoencoder,
        config: TrainingConfig,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.MSELoss()

        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.history = TrainingHistory()

        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> optim.Optimizer:
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, dataloader: DataLoader, epoch_bar=None) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        n_total = len(dataloader)

        for i, batch in enumerate(dataloader, 1):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            reconstructed = self.model(batch)
            loss = self.criterion(reconstructed, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if epoch_bar is not None and (
                i % max(1, n_total // 5) == 0 or i == n_total
            ):
                epoch_bar.set_postfix_str(
                    f"train batch {i}/{n_total}  loss={loss.item():.5f}"
                )

        return total_loss / n_batches

    def validate(self, dataloader: DataLoader, epoch_bar=None) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        n_total = len(dataloader)

        with torch.no_grad():
            for i, batch in enumerate(dataloader, 1):
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)

                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)

                total_loss += loss.item()
                n_batches += 1

                if epoch_bar is not None and (
                    i % max(1, n_total // 3) == 0 or i == n_total
                ):
                    epoch_bar.set_postfix_str(f"val batch {i}/{n_total}")

        if n_batches == 0:
            return 0.0

        return total_loss / n_batches

    def _update_live_plot(self, fig, axes, display_handle):
        epochs = range(1, len(self.history.train_loss) + 1)

        ax = axes[0]
        ax.clear()
        ax.plot(epochs, self.history.train_loss, label="Train", linewidth=1.5)
        ax.plot(epochs, self.history.val_loss, label="Val", linewidth=1.5)
        best_ep = int(np.argmin(self.history.val_loss)) + 1
        best_val = min(self.history.val_loss)
        ax.axvline(best_ep, color="grey", ls="--", alpha=0.4)
        ax.scatter([best_ep], [best_val], marker="*", s=100, color="red", zorder=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss  (best val={best_val:.5f} @ ep {best_ep})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.clear()
        ax.plot(epochs, self.history.learning_rates, color="tab:orange", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        display_handle.update(fig)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, live_plot: bool = True
    ) -> "TrainingHistory":
        patience_counter = 0
        has_validation = len(val_loader) > 0
        val_loss = float("inf")

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Training on {self.device}  |  Parameters: {n_params:,}")

        if not has_validation:
            print(
                "Warning: No validation data available. Using training loss for monitoring."
            )

        _do_live_plot = False
        fig, axes, _display_handle = None, None, None
        if live_plot and _HAS_IPYTHON:
            try:
                shell = get_ipython().__class__.__name__
                if shell in ("ZMQInteractiveShell", "Shell"):  # Jupyter / Colab
                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
                    plt.close(fig)  # prevent double render
                    # Create a display handle so we can update the plot in-place
                    # without clearing the tqdm bar
                    _display_handle = display(fig, display_id=True)
                    _do_live_plot = True
            except Exception:
                pass

        epoch_bar = tqdm(
            range(1, self.config.epochs + 1),
            desc="Epochs",
            unit="ep",
        )

        for epoch in epoch_bar:
            t0 = time.perf_counter()

            train_loss = self.train_epoch(train_loader, epoch_bar=epoch_bar)

            if has_validation:
                val_loss = self.validate(val_loader, epoch_bar=epoch_bar)
            else:
                val_loss = train_loss

            self.scheduler.step(val_loss)

            elapsed = time.perf_counter() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.update(train_loss, val_loss, current_lr, elapsed)

            is_best = val_loss < self.best_val_loss
            epoch_bar.set_postfix(
                train=f"{train_loss:.5f}",
                val=f"{val_loss:.5f}",
                lr=f"{current_lr:.1e}",
                best="★" if is_best else "",
            )
            if _do_live_plot:
                self._update_live_plot(fig, axes, _display_handle)
            if is_best:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pt", epoch, val_loss)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                epoch_bar.close()
                print(f"\nEarly stopping after epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    f"checkpoint_epoch_{epoch + 1}.pt", epoch, val_loss
                )

        self._save_checkpoint("final_model.pt", self.config.epochs, val_loss)
        self.history.save(self.config.checkpoint_dir / "training_history.json")

        print(f"\nDone — best val loss: {self.best_val_loss:.6f}")
        return self.history

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        self.checkpoint_manager.save(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            val_loss=val_loss,
            history=self.history.to_dict(),
        )

    @classmethod
    def from_simple_config(
        cls,
        data_path: str,
        model_type: str = "lightweight",
        window_size: int = 50,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        device: str = None,
    ) -> Tuple["AnomalyDetectionTrainer", AnomalyDataLoader, "TrainingHistory"]:
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

        config = TrainingConfig(
            learning_rate=learning_rate, epochs=epochs, early_stopping_patience=15
        )

        trainer = cls(model, config, device)

        print("\nStarting training...")
        history = trainer.fit(train_loader, val_loader)

        print("\nTraining complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")

        return trainer, data_loader, history


def train_anomaly_detector(
    data_path: str,
    model_type: str = "lightweight",
    window_size: int = 50,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = None,
) -> Tuple[BaseAutoencoder, AnomalyDataLoader, "TrainingHistory"]:
    """Convenience wrapper around AnomalyDetectionTrainer.from_simple_config()."""
    trainer, data_loader, history = AnomalyDetectionTrainer.from_simple_config(
        data_path=data_path,
        model_type=model_type,
        window_size=window_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    return trainer.model, data_loader, history


if __name__ == "__main__":
    model, data_loader, history = train_anomaly_detector(
        data_path="./data-output.json",
        model_type="lightweight",
        window_size=50,
        batch_size=16,
        epochs=100,
        learning_rate=1e-3,
    )
