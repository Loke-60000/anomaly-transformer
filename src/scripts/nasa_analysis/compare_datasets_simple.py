#!/usr/bin/env python3

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import os
import json


def compare_datasets():
    print("COMPARISON: NASA KAGGLE DATASET vs OUR TIMESERIES DATA")
    print("=" * 80)

    # Load NASA Kaggle dataset
    print("\n1. NASA KAGGLE DATASET STRUCTURE:")
    print("-" * 40)

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
        "labeled_anomalies.csv",
    )

    dataset_path = "/home/lokman/.cache/kagglehub/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/versions/1"

    # Analyze first channel in detail
    sample_channel = "P-1"
    train_data_nasa = np.load(f"{dataset_path}/data/data/train/{sample_channel}.npy")
    test_data_nasa = np.load(f"{dataset_path}/data/data/test/{sample_channel}.npy")

    print(f"Sample channel: {sample_channel}")
    print(f"Training data shape: {train_data_nasa.shape}")
    print(f"Test data shape: {test_data_nasa.shape}")
    print(f"Data type: {train_data_nasa.dtype}")
    print(f"Features per timestep: {train_data_nasa.shape[1]}")
    print(
        f"First feature range: [{train_data_nasa[:, 0].min():.3f}, {train_data_nasa[:, 0].max():.3f}]"
    )

    # Get anomaly info for this channel
    anomaly_info = df[df["chan_id"] == sample_channel].iloc[0]
    anomaly_sequences = eval(anomaly_info["anomaly_sequences"])
    print(f"Anomaly sequences: {anomaly_sequences}")

    print("\n2. OUR TIMESERIES DATA STRUCTURE:")
    print("-" * 40)

    # Check if we have our timeseries data
    timeseries_path = "timeseries-data/index.json"
    our_data = None

    if os.path.exists(timeseries_path):
        # Load our data using simple JSON loading
        try:
            with open(timeseries_path, "r") as f:
                index_data = json.load(f)

            # Try to load a sample node file
            for result in index_data["results"]:
                if result.get("bSuccess", False) and "sFilePath" in result:
                    node_path = os.path.join("timeseries-data", result["sFilePath"])
                    if os.path.exists(node_path):
                        with open(node_path, "r") as f:
                            node_data = json.load(f)

                        # Extract temperature data (unit 73)
                        curve_data = node_data["data"]["oCurveData"]["oData"]
                        if "73" in curve_data:
                            measurements = curve_data["73"]["mResult"]
                            sorted_timestamps = sorted(measurements.keys(), key=int)
                            values = [measurements[ts][0] for ts in sorted_timestamps]
                            our_data = np.array(values)

                            print(f"Our data shape: {our_data.shape}")
                            print(f"Data type: {our_data.dtype}")
                            print(f"Features per timestep: 1 (univariate)")
                            print(
                                f"Value range: [{our_data.min():.3f}, {our_data.max():.3f}]"
                            )
                            print(f"Length: {len(our_data)} timesteps")
                            break

        except Exception as e:
            print(f"Error loading our data: {e}")
            our_data = None
    else:
        print("No timeseries-data found in current directory")
        our_data = None

    print("\n3. KEY DIFFERENCES:")
    print("-" * 40)

    print("NASA Dataset:")
    print("  • Multivariate: 25 features per timestep")
    print("  • Labeled anomalies with exact start/end positions")
    print("  • 82 different channels (SMAP + MSL spacecraft)")
    print("  • Pre-normalized data")
    print("  • Train/test split already provided")
    print("  • Multiple types of anomalies per channel")

    print("\nOur Dataset:")
    print("  • Univariate: 1 feature per timestep (typically temperature)")
    print("  • No labeled anomalies (unsupervised learning)")
    print("  • Multiple sensor nodes")
    print("  • Raw sensor readings")
    print("  • Need to create train/test split")
    print("  • Unknown anomaly patterns")

    print("\n4. RECOMMENDATIONS FOR IMPROVEMENT:")
    print("-" * 40)

    print("Architecture improvements:")
    print("  1. Use the NASA dataset for supervised training/validation")
    print("  2. Adapt transformer for multivariate input (25 features)")
    print("  3. Implement anomaly sequence prediction capability")
    print("  4. Add contrastive learning for better representations")
    print("  5. Use attention visualization to understand anomaly patterns")

    print("\nModel architecture suggestions:")
    print("  1. Multi-head attention for different feature interactions")
    print("  2. Hierarchical attention (feature-level + temporal-level)")
    print("  3. Variational autoencoder for uncertainty estimation")
    print("  4. Memory-augmented networks for rare anomaly patterns")
    print("  5. Ensemble of models trained on different channels")

    return {
        "nasa_train_shape": train_data_nasa.shape,
        "nasa_test_shape": test_data_nasa.shape,
        "our_data_shape": our_data.shape if our_data is not None else None,
        "nasa_features": train_data_nasa.shape[1],
        "our_features": 1,
        "nasa_channels": len(df),
        "nasa_total_anomalies": sum(
            len(eval(row["anomaly_sequences"])) for _, row in df.iterrows()
        ),
    }


def suggest_improved_architecture():
    print("\n5. IMPROVED TRANSFORMER ARCHITECTURE DESIGN:")
    print("=" * 60)

    architecture_code = '''
class ImprovedTransformerAutoencoder(nn.Module):
    """
    Enhanced transformer autoencoder for multivariate time series anomaly detection.
    Features:
    - Multi-head attention for feature interactions
    - Hierarchical encoding (feature + temporal)
    - Variational bottleneck for uncertainty
    - Attention visualization
    """
    
    def __init__(
        self,
        input_dim: int = 25,  # NASA dataset has 25 features
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 32,
        use_variational: bool = True,
    ):
        super().__init__()
        
        # Feature-wise embedding
        self.feature_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Multi-scale transformer blocks
        self.local_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead//2, dim_feedforward//2, dropout),
            num_layers=num_layers//2
        )
        
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_layers//2
        )
        
        # Variational bottleneck
        if use_variational:
            self.mu_layer = nn.Linear(d_model, latent_dim)
            self.logvar_layer = nn.Linear(d_model, latent_dim)
            self.z_to_hidden = nn.Linear(latent_dim, d_model)
        else:
            self.bottleneck = nn.Linear(d_model, latent_dim)
            self.expand = nn.Linear(latent_dim, d_model)
        
        # Decoder with cross attention
        self.decoder_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers=num_layers
        )
        
        # Output projection with residual connection
        self.output_projection = nn.Linear(d_model, input_dim)
        self.residual_gate = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Store input for residual connection
        input_residual = x
        
        # Transpose for transformer: [seq_len, batch, features]
        x = x.transpose(0, 1)
        
        # Embed features
        x = self.feature_embedding(x)
        x = self.pos_encoder(x)
        
        # Multi-scale encoding
        local_features = self.local_transformer(x)
        global_features = self.global_transformer(local_features)
        
        # Variational bottleneck for uncertainty
        if hasattr(self, 'mu_layer'):
            mu = self.mu_layer(global_features)
            logvar = self.logvar_layer(global_features)
            z = self.reparameterize(mu, logvar)
            decoded_input = self.z_to_hidden(z)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        else:
            compressed = self.bottleneck(global_features)
            decoded_input = self.expand(compressed)
            kld_loss = torch.tensor(0.0, device=x.device)
        
        # Decode with cross-attention
        output = self.decoder_transformer(decoded_input, global_features)
        output = self.output_projection(output)
        
        # Transpose back: [batch, seq_len, features]
        output = output.transpose(0, 1)
        
        # Residual connection with gating
        gate = torch.sigmoid(self.residual_gate(input_residual))
        output = gate * output + (1 - gate) * input_residual
        
        return output, kld_loss
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
'''

    print(architecture_code)

    training_suggestions = """
TRAINING STRATEGY SUGGESTIONS:

1. Multi-Stage Training:
   Stage 1: Pre-train on NASA dataset (supervised with labeled anomalies)
   Stage 2: Fine-tune on our data (unsupervised adaptation)
   Stage 3: Ensemble different models for robust detection

2. Enhanced Loss Function:
   Total Loss = α*Reconstruction_Loss + β*KL_Divergence + γ*Contrastive_Loss + δ*Anomaly_Classification_Loss
   
   Where:
   - Reconstruction_Loss: MSE between input and output
   - KL_Divergence: Regularization for variational component
   - Contrastive_Loss: Encourages normal samples to cluster, anomalies to spread
   - Anomaly_Classification_Loss: Direct supervision from NASA labels

3. Data Augmentation Techniques:
   - Temporal masking (mask random time windows)
   - Feature dropout (randomly zero out features)
   - Gaussian noise injection (add controlled noise)
   - Time warping (stretch/compress time sequences)
   - Mixup between normal sequences

4. Advanced Evaluation Metrics:
   - Point-wise anomaly detection (ROC-AUC, PR-AUC)
   - Sequence-wise anomaly detection (label entire sequences)
   - Early anomaly detection (detect anomalies as they start)
   - Attention pattern consistency (visualize what model focuses on)

5. Anomaly Scoring Methods:
   - Reconstruction error (L1/L2 distance)
   - Latent space Mahalanobis distance
   - Attention entropy (high entropy = anomaly)
   - Ensemble disagreement (variance across models)
   - Temporal smoothness violations
"""

    print(training_suggestions)


if __name__ == "__main__":
    comparison_results = compare_datasets()
    suggest_improved_architecture()

    print(f"\nSUMMARY STATISTICS:")
    print(
        f"NASA dataset: {comparison_results['nasa_channels']} channels, {comparison_results['nasa_total_anomalies']} total anomalies"
    )
    print(
        f"Feature dimensions: NASA={comparison_results['nasa_features']}, Ours={comparison_results['our_features']}"
    )
    print(f"This represents a significant upgrade opportunity!")
    print(f"The NASA dataset provides:")
    print(f"  • Rich multivariate features (25D vs 1D)")
    print(f"  • Labeled ground truth for supervised learning")
    print(f"  • Multiple spacecraft domains for transfer learning")
    print(f"  • Established benchmarks for comparison")
