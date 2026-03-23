import torch
import torch.nn as nn
import math
from typing import Optional
from abc import ABC, abstractmethod

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class BaseAutoencoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        pass

    def get_reconstruction_error(
        self, src: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        reconstructed = self.forward(src)
        error = torch.abs(src - reconstructed)

        if reduction == "mean":
            return error.mean(dim=(1, 2))
        elif reduction == "sum":
            return error.sum(dim=(1, 2))
        else:
            return error

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerAutoencoder(BaseAutoencoder):
    """Full encoder-decoder transformer autoencoder.

    The decoder uses **learned query tokens** instead of the raw input, so
    reconstruction must pass through the encoder's compressed memory —
    creating a genuine information bottleneck.
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
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, input_dim)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Learned decoder queries — the decoder cannot see the raw input.
        # Shape (max_seq_len, d_model); sliced to actual seq_len at runtime.
        self.decoder_queries = nn.Parameter(
            torch.randn(max_seq_len, d_model) * 0.02
        )
        self.decoder_pos = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = self._build_encoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation
        )
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
        # src: (batch, seq_len, input_dim)
        seq_len, batch_size = src.size(1), src.size(0)

        src_t = src.transpose(0, 1)                              # (S, B, F)
        projected = self.pos_encoder(self.input_projection(src_t))  # (S, B, D)
        memory = self.encoder(projected, src_mask)                # (S, B, D)

        # Learned queries — no access to raw input
        tgt = self.decoder_queries[:seq_len].unsqueeze(1).expand(-1, batch_size, -1)
        tgt = self.decoder_pos(tgt)                               # (S, B, D)

        output = self.decoder(tgt, memory)                        # (S, B, D)
        return self.output_projection(output).transpose(0, 1)    # (B, S, F)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src = src.transpose(0, 1)
        src = self.pos_encoder(self.input_projection(src))
        return self.encoder(src).transpose(0, 1)


class LightweightTransformerAutoencoder(BaseAutoencoder):
    """Shared-encoder transformer with a configurable bottleneck.

    The ``bottleneck_ratio`` controls how aggressively the latent space is
    compressed.  A ratio of 4 (default) maps ``d_model → d_model/4 → d_model``,
    giving a much tighter information bottleneck than the previous ``d_model/2``.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        bottleneck_ratio: int = 4,
    ):
        super().__init__()

        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, input_dim)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = self._build_transformer(
            d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.bottleneck = self._build_bottleneck(d_model, bottleneck_ratio)

        self._init_weights()

    def _build_transformer(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_bottleneck(self, d_model: int, ratio: int) -> nn.Sequential:
        latent = max(d_model // ratio, 4)  # floor at 4 dims
        return nn.Sequential(
            nn.Linear(d_model, latent),
            nn.GELU(),
            nn.LayerNorm(latent),
            nn.Linear(latent, d_model),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = src.transpose(0, 1)
        x = self.pos_encoder(self.input_projection(src))
        encoded = self.transformer(x)
        compressed = self.bottleneck(encoded)
        return self.output_projection(compressed).transpose(0, 1)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src = src.transpose(0, 1)
        x = self.pos_encoder(self.input_projection(src))
        encoded = self.transformer(x)
        return self.bottleneck(encoded).transpose(0, 1)

def create_model(model_type: str = "lightweight", **kwargs) -> BaseAutoencoder:
    models = {
        "lightweight": LightweightTransformerAutoencoder,
        "full": TransformerAutoencoder,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )

    return models[model_type](**kwargs)
