"""
Basic transformer models for anomaly detection which provides
enhanced multivariate anomaly detection with:
    - Multi-scale positional encoding
    - Feature attention mechanisms
    - Hierarchical encoding for complex patterns
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from abc import ABC, abstractmethod


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of model embeddings
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class BaseAutoencoder(nn.Module, ABC):
    """Abstract base class for autoencoder models."""

    @abstractmethod
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        pass

    @abstractmethod
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        pass

    def get_reconstruction_error(
        self, src: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Calculate reconstruction error for anomaly detection.

        Args:
            src: Input tensor
            reduction: 'none', 'mean', or 'sum'

        Returns:
            Reconstruction error
        """
        reconstructed = self.forward(src)
        error = torch.abs(src - reconstructed)

        if reduction == "mean":
            return error.mean(dim=(1, 2))
        elif reduction == "sum":
            return error.sum(dim=(1, 2))
        else:
            return error

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerAutoencoder(BaseAutoencoder):
    """
    Full transformer autoencoder with separate encoder and decoder.
    Suitable for complex time series patterns.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim

        # Projection layers
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, input_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Encoder
        self.encoder = self._build_encoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation
        )

        # Decoder
        self.decoder = self._build_decoder(
            d_model, nhead, num_decoder_layers, dim_feedforward, dropout, activation
        )

        self._init_weights()

    def _build_encoder(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> nn.TransformerEncoder:
        """Build transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_decoder(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> nn.TransformerDecoder:
        """Build transformer decoder."""
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
        )
        return nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through autoencoder.

        Args:
            src: Input tensor [batch_size, seq_len, input_dim]
            src_mask: Optional attention mask

        Returns:
            Reconstructed tensor [batch_size, seq_len, input_dim]
        """
        # Transpose to [seq_len, batch_size, input_dim]
        src = src.transpose(0, 1)

        # Project and add positional encoding
        src = self.input_projection(src)
        src = self.pos_encoder(src)

        # Encode
        memory = self.encoder(src, src_mask)

        # Decode
        output = self.decoder(src, memory)

        # Project back to input dimension
        output = self.output_projection(output)

        # Transpose back to [batch_size, seq_len, input_dim]
        return output.transpose(0, 1)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        src = src.transpose(0, 1)
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src)
        return memory.transpose(0, 1)


class LightweightTransformerAutoencoder(BaseAutoencoder):
    """
    Lightweight transformer autoencoder with shared encoder.
    Suitable for smaller datasets and faster training.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Projection layers
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, input_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Single transformer encoder
        self.transformer = self._build_transformer(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )

        # Information bottleneck
        self.bottleneck = self._build_bottleneck(d_model)

        self._init_weights()

    def _build_transformer(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> nn.TransformerEncoder:
        """Build transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_bottleneck(self, d_model: int) -> nn.Sequential:
        """Build bottleneck layer for dimensionality reduction."""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.

        Args:
            src: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Reconstructed tensor [batch_size, seq_len, input_dim]
        """
        # Transpose to [seq_len, batch_size, input_dim]
        src = src.transpose(0, 1)

        # Project and encode
        x = self.input_projection(src)
        x = self.pos_encoder(x)

        # Transform
        encoded = self.transformer(x)

        # Apply bottleneck (creates information constraint for anomaly detection)
        compressed = self.bottleneck(encoded)

        # Decode
        output = self.output_projection(compressed)

        # Transpose back to [batch_size, seq_len, input_dim]
        return output.transpose(0, 1)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        src = src.transpose(0, 1)
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        compressed = self.bottleneck(encoded)
        return compressed.transpose(0, 1)


def create_model(model_type: str = "lightweight", **kwargs) -> BaseAutoencoder:
    """
    Factory function to create autoencoder models.

    Args:
        model_type: 'lightweight' or 'full'
        **kwargs: Model configuration parameters

    Returns:
        Autoencoder model instance
    """
    models = {
        "lightweight": LightweightTransformerAutoencoder,
        "full": TransformerAutoencoder,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )

    return models[model_type](**kwargs)
