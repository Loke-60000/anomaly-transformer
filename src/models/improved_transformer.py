"""
Enhanced Transformer Autoencoder for Multivariate Time Series Anomaly Detection.
Designed to work with NASA SMAP/MSL dataset and provides significant improvements
over the basic transformer architecture.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePositionalEncoding(nn.Module):
    """Enhanced positional encoding with multiple frequency scales."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create multiple positional encodings at different scales
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()

        # Standard encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add slower frequency components for long-term patterns
        if d_model >= 32:
            slow_div_term = torch.exp(
                torch.arange(0, min(d_model // 4, 16), 2).float()
                * (-math.log(100000.0) / (d_model // 4))
            )
            pe[:, : len(slow_div_term)] += 0.2 * torch.sin(position * slow_div_term)

        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    """Attention mechanism for feature interactions."""

    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim // 2

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [seq_len, batch, features]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores across features
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(attention_weights, V)
        output = self.output(attended)

        return output, attention_weights


class VAEBottleneck(nn.Module):
    """Variational bottleneck for uncertainty estimation."""

    def __init__(self, input_dim: int, latent_dim: int, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
        )

        self.mu_layer = nn.Linear(input_dim // 4, latent_dim)
        self.logvar_layer = nn.Linear(input_dim // 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [seq_len, batch, features]
        batch_size = x.size(1)

        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)

        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (batch_size * x.size(0))  # Normalize by batch and sequence

        return decoded, self.beta * kl_loss, mu, logvar


class LightningIndexer(nn.Module):
    """
    DeepSeek Lightning Indexer for Sparse Attention.
    Computes index scores to select top-k relevant tokens.
    """

    def __init__(self, d_model: int, d_index: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.d_index = d_index

        # Projections for Query, Key, and Weight
        self.q_proj = nn.Linear(d_model, num_heads * d_index)
        self.k_proj = nn.Linear(d_model, d_index)  # Shared key across heads (MQA style)
        self.w_proj = nn.Linear(d_model, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        seq_len, batch_size, _ = x.size()

        # Project and reshape
        # Permute to [batch, seq_len, ...] for easier calculation
        x_batch = x.permute(1, 0, 2)  # [batch, seq_len, d_model]

        q = self.q_proj(x_batch).view(
            batch_size, seq_len, self.num_heads, self.d_index
        )  # [B, T, H, D_I]
        k = self.k_proj(x_batch).view(batch_size, seq_len, self.d_index)  # [B, T, D_I]
        w = self.w_proj(x_batch).view(batch_size, seq_len, self.num_heads)  # [B, T, H]

        # Compute Index Scores
        # I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s)

        # 1. Dot product q_{t,j} . k_s
        # We want a score for every pair (t, s).
        # q: [B, T_q, H, D_I]
        # k: [B, T_k, D_I]
        # Result: [B, T_q, T_k, H]
        # Using einsum: 'bthd, bsd -> btsh' (t=query time, s=key time)
        dot_product = torch.einsum("bthd, bsd -> btsh", q, k)

        # 2. ReLU
        activated = F.relu(dot_product)

        # 3. Weighted sum
        # w: [B, T_q, H] -> [B, T_q, 1, H]
        # Result: [B, T_q, T_k]
        index_scores = torch.sum(w.unsqueeze(2) * activated, dim=-1)

        return index_scores


class DSATransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer equipped with DeepSeek Sparse Attention (DSA).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        dsa_top_k: int = 32,
        dsa_d_index: int = 32,
        dsa_num_heads: int = 4,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.indexer = LightningIndexer(
            d_model, d_index=dsa_d_index, num_heads=dsa_num_heads
        )
        self.dsa_top_k = dsa_top_k

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.norm_first = norm_first

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # src: [seq_len, batch, d_model] (since batch_first=False by default in this project)

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # DeepSeek Sparse Attention Logic
        # 1. Compute Index Scores
        # index_scores: [batch, seq_len, seq_len]
        index_scores = self.indexer(x)

        # 2. Create Sparse Mask
        # Select top-k indices for each query
        # If sequence length is smaller than top_k, we select all (dense attention)
        seq_len = x.size(0)
        k = min(self.dsa_top_k, seq_len)

        # Top-k selection
        # values, indices: [batch, seq_len, k]
        _, topk_indices = torch.topk(index_scores, k, dim=-1)

        # Create a mask [batch, seq_len, seq_len]
        # Initialize with -inf
        batch_size = x.size(1)
        sparse_mask = torch.full(
            (batch_size, seq_len, seq_len),
            float("-inf"),
            device=x.device,
            dtype=x.dtype,
        )

        # Scatter 0.0 to selected indices
        # We need to construct indices for scatter
        # dim 0: batch indices (broadcasted)
        # dim 1: query indices (broadcasted)
        # dim 2: key indices (from topk_indices)

        # This scatter is a bit tricky in 3D.
        # Let's use scatter_ on the last dimension.
        sparse_mask.scatter_(2, topk_indices, 0.0)

        # Combine with existing attn_mask if provided
        if attn_mask is not None:
            # attn_mask usually [seq_len, seq_len] or [batch*nhead, ...]
            # If it's [seq_len, seq_len], we broadcast
            sparse_mask = sparse_mask + attn_mask

        # Prepare mask for MHA
        # MHA expects [batch * num_heads, seq_len, seq_len] for 3D mask
        num_heads = self.self_attn.num_heads
        # Repeat mask for each head
        # [batch, seq_len, seq_len] -> [batch, num_heads, seq_len, seq_len] -> [batch*num_heads, seq_len, seq_len]
        sparse_mask_expanded = (
            sparse_mask.unsqueeze(1)
            .expand(-1, num_heads, -1, -1)
            .reshape(batch_size * num_heads, seq_len, seq_len)
        )

        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=sparse_mask_expanded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class ImprovedTransformerAutoencoder(nn.Module):
    """
    Enhanced transformer autoencoder for multivariate time series anomaly detection.

    Key improvements:
    - Multi-scale positional encoding
    - Feature attention for cross-feature interactions
    - Hierarchical encoder (local + global patterns)
    - Variational bottleneck for uncertainty estimation
    - Residual connections with gating
    - Attention visualization support
    """

    def __init__(
        self,
        input_dim: int = 25,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 32,
        use_variational: bool = True,
        use_feature_attention: bool = True,
        beta: float = 1.0,
        max_sequence_length: int = 1000,
        use_dsa: bool = False,
        dsa_top_k: int = 32,
        dsa_d_index: int = 32,
        dsa_num_heads: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_variational = use_variational
        self.use_feature_attention = use_feature_attention
        self.max_sequence_length = max_sequence_length
        self.use_dsa = use_dsa

        # Feature embedding layers
        self.input_projection = nn.Linear(input_dim, d_model)
        self.feature_norm = nn.LayerNorm(d_model)

        # Enhanced positional encoding
        self.pos_encoder = MultiScalePositionalEncoding(
            d_model, max_sequence_length, dropout
        )

        # Feature attention (cross-feature interactions)
        if use_feature_attention:
            self.feature_attention = FeatureAttention(input_dim)

        # Multi-scale transformer encoding
        # Helper to create encoder layer
        def create_encoder_layer(nhead_val, dim_feedforward_val):
            if use_dsa:
                return DSATransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead_val,
                    dim_feedforward=dim_feedforward_val,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=True,
                    dsa_top_k=dsa_top_k,
                    dsa_d_index=dsa_d_index,
                    dsa_num_heads=dsa_num_heads,
                )
            else:
                return nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead_val,
                    dim_feedforward=dim_feedforward_val,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=False,
                    norm_first=True,
                )

        # Local patterns (short-term dependencies)
        self.local_encoder = nn.TransformerEncoder(
            create_encoder_layer(max(1, nhead // 2), dim_feedforward // 2),
            num_layers=max(1, num_layers // 2),
            norm=nn.LayerNorm(d_model),
        )

        # Global patterns (long-term dependencies)
        self.global_encoder = nn.TransformerEncoder(
            create_encoder_layer(nhead, dim_feedforward),
            num_layers=max(1, num_layers // 2),
            norm=nn.LayerNorm(d_model),
        )

        # Bottleneck (variational or standard)
        if use_variational:
            self.bottleneck = VAEBottleneck(d_model, latent_dim, beta)
        else:
            self.bottleneck = nn.Sequential(
                nn.Linear(d_model, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, d_model),
            )

        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=False,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection with residual gating
        self.output_projection = nn.Linear(d_model, input_dim)
        self.residual_gate = nn.Linear(input_dim, input_dim)
        self.output_norm = nn.LayerNorm(input_dim)

        # Storage for attention weights (for visualization)
        self.attention_weights: Dict[str, torch.Tensor] = {}

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Tuple of (reconstructed_x, attention_weights, mu, logvar)
        """
        batch_size, seq_len, features = x.shape
        losses = {}

        # Store input for residual connection
        input_residual = x

        # Feature attention (if enabled)
        if self.use_feature_attention:
            # Apply across time dimension
            x_feat_attended = []
            feature_attention_weights = []

            for t in range(seq_len):
                x_t = x[:, t : t + 1, :].transpose(0, 1)  # [1, batch, features]
                attended, attn_weights = self.feature_attention(x_t)
                x_feat_attended.append(attended.transpose(0, 1))
                feature_attention_weights.append(attn_weights.squeeze(0))

            x = torch.cat(x_feat_attended, dim=1)
            self.attention_weights["feature_attention"] = torch.stack(
                feature_attention_weights, dim=1
            )

        # Transpose for transformer: [seq_len, batch, features]
        x = x.transpose(0, 1)

        # Project to model dimension
        x = self.input_projection(x)
        x = self.feature_norm(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Multi-scale encoding
        local_features = self.local_encoder(x)
        global_features = self.global_encoder(local_features)

        # Apply bottleneck
        if self.use_variational:
            compressed, kl_loss, mu, logvar = self.bottleneck(global_features)
            losses["kl_divergence"] = kl_loss
        else:
            compressed = self.bottleneck(global_features)
            losses["kl_divergence"] = torch.tensor(0.0, device=x.device)
            mu = torch.zeros_like(compressed)
            logvar = torch.zeros_like(compressed)

        # Decode
        decoded = self.decoder(compressed, global_features)

        # Project back to input dimension
        output = self.output_projection(decoded)

        # Transpose back: [batch, seq_len, features]
        output = output.transpose(0, 1)

        # Residual connection with learnable gating
        gate = torch.sigmoid(self.residual_gate(input_residual))
        output = gate * output + (1 - gate) * input_residual
        output = self.output_norm(output)

        return output, self.attention_weights, mu, logvar

    def kl_divergence_loss(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate KL divergence loss.

        Args:
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            KL divergence loss
        """
        # KL divergence = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize by batch size and sequence length
        kl_loss = kl_loss / (mu.size(0) * mu.size(1))
        return kl_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        x = x.transpose(0, 1)
        x = self.input_projection(x)
        x = self.feature_norm(x)
        x = self.pos_encoder(x)

        local_features = self.local_encoder(x)
        global_features = self.global_encoder(local_features)

        if self.use_variational:
            compressed, _ = self.bottleneck(global_features)
        else:
            compressed = self.bottleneck(global_features)

        return compressed.transpose(0, 1)

    def get_anomaly_scores(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multiple anomaly scores.

        Args:
            x: Input tensor [batch, seq_len, features]
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Dictionary with different anomaly score types
        """
        with torch.no_grad():
            reconstructed, attention_weights, mu, logvar = self.forward(x)

            # Reconstruction error (L1 and L2)
            l1_error = torch.abs(x - reconstructed)
            l2_error = torch.pow(x - reconstructed, 2)

            # Feature-wise errors
            feature_l1 = l1_error.sum(dim=1)  # [batch, features]
            feature_l2 = l2_error.sum(dim=1)  # [batch, features]

            # Temporal errors
            temporal_l1 = l1_error.sum(dim=2)  # [batch, seq_len]
            temporal_l2 = l2_error.sum(dim=2)  # [batch, seq_len]

            # Overall errors
            if reduction == "mean":
                total_l1 = l1_error.mean(dim=(1, 2))  # [batch]
                total_l2 = l2_error.mean(dim=(1, 2))  # [batch]
            elif reduction == "sum":
                total_l1 = l1_error.sum(dim=(1, 2))  # [batch]
                total_l2 = l2_error.sum(dim=(1, 2))  # [batch]
            else:
                total_l1 = l1_error  # [batch, seq_len, features]
                total_l2 = l2_error  # [batch, seq_len, features]

            # Calculate KL divergence for scoring
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            # Average over sequence length if needed, or keep as [batch, seq_len]
            # Here we'll take mean over sequence length for a per-sample score
            kl_score = kl_loss.mean(dim=1)

            scores = {
                "reconstruction_l1": total_l1,
                "reconstruction_l2": total_l2,
                "feature_l1": feature_l1,
                "feature_l2": feature_l2,
                "temporal_l1": temporal_l1,
                "temporal_l2": temporal_l2,
                "kl_divergence": kl_score,
            }

            # Add attention-based scores if available
            if "feature_attention" in self.attention_weights:
                # High attention entropy might indicate anomalies
                attn = self.attention_weights[
                    "feature_attention"
                ]  # [batch, seq_len, features, features]
                entropy = -(attn * torch.log(attn + 1e-8)).sum(
                    dim=-1
                )  # [batch, seq_len, features]
                if entropy.dim() >= 3:
                    scores["attention_entropy"] = entropy.mean(
                        dim=-1
                    )  # [batch, seq_len]
                else:
                    scores["attention_entropy"] = (
                        entropy.mean(dim=-1) if entropy.dim() > 1 else entropy
                    )

        return scores

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Return stored attention weights for visualization."""
        return self.attention_weights.copy()


class AnomalyScoreCalculator:
    """
    Calculator for comprehensive anomaly scores using multiple metrics.
    """

    def __init__(self):
        pass

    def calculate_comprehensive_scores(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        attention_weights: Dict[str, torch.Tensor],
        latent_mean: torch.Tensor,
        latent_logvar: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate multiple anomaly scores and combine them.

        Args:
            original: Original input tensor [batch, seq_len, features]
            reconstructed: Reconstructed tensor [batch, seq_len, features]
            attention_weights: Dictionary of attention weights
            latent_mean: Latent mean tensor
            latent_logvar: Latent log variance tensor

        Returns:
            Dictionary of score arrays
        """
        batch_size = original.shape[0]

        # 1. Reconstruction Error (MSE)
        # [batch, seq_len, features] -> [batch]
        recon_error = torch.mean((original - reconstructed) ** 2, dim=(1, 2))

        # 2. KL Divergence Score (approximate)
        # High KL divergence indicates the sample is far from the prior distribution
        kl_term = -0.5 * (
            1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()
        )
        latent_dim = latent_mean.shape[-1] if latent_mean.dim() > 0 else 1
        if kl_term.dim() == 3:
            # [seq_len, batch, latent_dim] -> per-sample score
            kl_score = kl_term.sum(dim=-1).mean(dim=0) / latent_dim
        elif kl_term.dim() == 2:
            # [batch, latent_dim]
            kl_score = kl_term.sum(dim=-1) / latent_dim
        else:
            # Fallback to averaging remaining dimensions
            kl_score = kl_term.view(batch_size, -1).mean(dim=1) / latent_dim

        # 3. Attention Entropy (if available)
        # Low entropy means focused attention (normal), high entropy means scattered (potential anomaly)
        attn_score = torch.zeros(batch_size, device=original.device)
        if "feature_attention" in attention_weights:
            attn = attention_weights["feature_attention"]
            # Older checkpoints stored [batch, seq_len, features, features]
            # Newer runs (for CPU-friendly script) use [batch, seq_len, features]
            if attn.dim() == 4:
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
                attn_score = torch.mean(entropy, dim=(1, 2))
            elif attn.dim() == 3:
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
                attn_score = torch.mean(entropy, dim=1)
            else:
                # Fallback: flatten remaining dims per sample
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1)
                attn_score = entropy.view(batch_size, -1).mean(dim=1)

        # Combine scores (simple weighted sum for now, can be tuned)
        # Normalize scores to be roughly in the same range before combining
        # This is a heuristic; in production, you'd fit a distribution to normal data scores

        # Convert to numpy for return
        recon_np = recon_error.detach().cpu().numpy()
        kl_np = kl_score.detach().cpu().numpy()
        attn_np = attn_score.detach().cpu().numpy()

        # Simple combination: Reconstruction is usually the strongest signal
        combined_score = recon_np + 0.1 * kl_np + 0.01 * attn_np

        return {
            "reconstruction": recon_np,
            "kl_divergence": kl_np,
            "attention": attn_np,
            "combined": combined_score,
        }


def create_improved_model(config: Dict) -> ImprovedTransformerAutoencoder:
    """Factory function to create improved transformer autoencoder."""
    return ImprovedTransformerAutoencoder(
        input_dim=config.get("input_dim", 25),
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 8),
        num_layers=config.get("num_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 512),
        dropout=config.get("dropout", 0.1),
        latent_dim=config.get("latent_dim", 32),
        use_variational=config.get("use_variational", True),
        use_feature_attention=config.get("use_feature_attention", True),
        beta=config.get("beta", 1.0),
        max_sequence_length=config.get("max_sequence_length", 1000),
        use_dsa=config.get("use_dsa", False),
        dsa_top_k=config.get("dsa_top_k", 32),
        dsa_d_index=config.get("dsa_d_index", 32),
        dsa_num_heads=config.get("dsa_num_heads", 4),
    )


# Example configuration for NASA SMAP/MSL dataset
NASA_CONFIG = {
    "input_dim": 25,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "latent_dim": 32,
    "use_variational": True,
    "use_feature_attention": True,
    "beta": 1.0,
    "max_sequence_length": 10000,
}
