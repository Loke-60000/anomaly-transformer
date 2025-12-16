"""
Unit tests for transformer models.
"""

import pytest
import torch
import torch.nn as nn

import sys

sys.path.append("..")

from src.models.model import (
    PositionalEncoding,
    BaseAutoencoder,
    TransformerAutoencoder,
    LightweightTransformerAutoencoder,
    create_model,
)


class TestPositionalEncoding:
    """Test PositionalEncoding module."""

    def test_positional_encoding_creation(self):
        """Test basic positional encoding creation."""
        d_model = 32
        pos_enc = PositionalEncoding(d_model=d_model, max_len=100, dropout=0.1)

        assert pos_enc.pe.shape[2] == d_model
        assert pos_enc.pe.shape[0] == 100

    def test_positional_encoding_forward(self):
        """Test forward pass through positional encoding."""
        d_model = 32
        seq_len = 50
        batch_size = 4

        pos_enc = PositionalEncoding(d_model=d_model, dropout=0.0)

        # Input: [seq_len, batch_size, d_model]
        x = torch.randn(seq_len, batch_size, d_model)
        output = pos_enc(x)

        # Output shape should match input shape
        assert output.shape == x.shape

    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        d_model = 32
        seq_len = 50
        batch_size = 4

        pos_enc = PositionalEncoding(d_model=d_model, dropout=0.0)
        pos_enc.eval()  # Set to eval mode to disable dropout

        x = torch.randn(seq_len, batch_size, d_model)
        output1 = pos_enc(x)
        output2 = pos_enc(x)

        torch.testing.assert_close(output1, output2)


class TestTransformerAutoencoder:
    """Test TransformerAutoencoder model."""

    def test_model_creation(self):
        """Test basic model creation."""
        model = TransformerAutoencoder(
            input_dim=1,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )

        assert model.d_model == 64
        assert model.input_dim == 1
        assert isinstance(model, BaseAutoencoder)

    def test_model_forward(self):
        """Test forward pass."""
        batch_size = 8
        seq_len = 50
        input_dim = 1

        model = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=32,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )

        # Input: [batch_size, seq_len, input_dim]
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_model_encode(self):
        """Test encoding."""
        batch_size = 8
        seq_len = 50
        input_dim = 1
        d_model = 32

        model = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )

        x = torch.randn(batch_size, seq_len, input_dim)
        encoded = model.encode(x)

        # Encoded should be [batch_size, seq_len, d_model]
        assert encoded.shape == (batch_size, seq_len, d_model)

    def test_model_reconstruction_error(self):
        """Test reconstruction error calculation."""
        batch_size = 8
        seq_len = 50
        input_dim = 1

        model = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=32,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )

        x = torch.randn(batch_size, seq_len, input_dim)

        # Test different reductions
        error_mean = model.get_reconstruction_error(x, reduction="mean")
        error_sum = model.get_reconstruction_error(x, reduction="sum")
        error_none = model.get_reconstruction_error(x, reduction="none")

        assert error_mean.shape == (batch_size,)
        assert error_sum.shape == (batch_size,)
        assert error_none.shape == x.shape

    def test_model_gradient_flow(self):
        """Test that gradients flow properly."""
        model = TransformerAutoencoder(
            input_dim=1, d_model=32, nhead=2, num_encoder_layers=2, num_decoder_layers=2
        )

        x = torch.randn(4, 20, 1, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None


class TestLightweightTransformerAutoencoder:
    """Test LightweightTransformerAutoencoder model."""

    def test_model_creation(self):
        """Test basic model creation."""
        model = LightweightTransformerAutoencoder(
            input_dim=1,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )

        assert model.d_model == 32
        assert isinstance(model, BaseAutoencoder)

    def test_model_forward(self):
        """Test forward pass."""
        batch_size = 8
        seq_len = 50
        input_dim = 1

        model = LightweightTransformerAutoencoder(
            input_dim=input_dim, d_model=32, nhead=2, num_layers=2
        )

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_model_encode(self):
        """Test encoding."""
        batch_size = 8
        seq_len = 50
        input_dim = 1
        d_model = 32

        model = LightweightTransformerAutoencoder(
            input_dim=input_dim, d_model=d_model, nhead=2, num_layers=2
        )

        x = torch.randn(batch_size, seq_len, input_dim)
        encoded = model.encode(x)

        # Encoded should be [batch_size, seq_len, d_model]
        assert encoded.shape == (batch_size, seq_len, d_model)

    def test_bottleneck_compression(self):
        """Test that bottleneck compresses information."""
        model = LightweightTransformerAutoencoder(
            input_dim=1, d_model=32, nhead=2, num_layers=2
        )

        # The bottleneck should compress from d_model -> d_model//2 -> d_model
        assert model.bottleneck[0].out_features == 16  # d_model // 2
        assert model.bottleneck[2].out_features == 32  # d_model

    def test_model_reconstruction_error(self):
        """Test reconstruction error calculation."""
        batch_size = 8
        seq_len = 50
        input_dim = 1

        model = LightweightTransformerAutoencoder(
            input_dim=input_dim, d_model=32, nhead=2, num_layers=2
        )

        x = torch.randn(batch_size, seq_len, input_dim)
        error = model.get_reconstruction_error(x, reduction="mean")

        assert error.shape == (batch_size,)
        assert (error >= 0).all()  # Errors should be non-negative


class TestModelFactory:
    """Test create_model factory function."""

    def test_create_lightweight_model(self):
        """Test creating lightweight model."""
        model = create_model(
            model_type="lightweight", input_dim=1, d_model=32, nhead=2, num_layers=2
        )

        assert isinstance(model, LightweightTransformerAutoencoder)
        assert model.d_model == 32

    def test_create_full_model(self):
        """Test creating full model."""
        model = create_model(
            model_type="full",
            input_dim=1,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
        )

        assert isinstance(model, TransformerAutoencoder)
        assert model.d_model == 64

    def test_create_invalid_model(self):
        """Test creating model with invalid type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(model_type="invalid")

    def test_factory_with_default_args(self):
        """Test factory with minimal arguments."""
        model = create_model(model_type="lightweight")
        assert isinstance(model, LightweightTransformerAutoencoder)


class TestModelComparison:
    """Test comparison between full and lightweight models."""

    def test_model_size_difference(self):
        """Test that lightweight model is smaller."""
        full_model = create_model(
            model_type="full",
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
        )

        lightweight_model = create_model(
            model_type="lightweight", d_model=32, nhead=2, num_layers=2
        )

        full_params = sum(p.numel() for p in full_model.parameters())
        lightweight_params = sum(p.numel() for p in lightweight_model.parameters())

        # Lightweight should have fewer parameters
        assert lightweight_params < full_params

    def test_both_models_work_same_input(self):
        """Test that both models can process same input."""
        x = torch.randn(4, 50, 1)

        full_model = create_model(model_type="full")
        lightweight_model = create_model(model_type="lightweight")

        # Both should produce output of same shape
        full_output = full_model(x)
        lightweight_output = lightweight_model(x)

        assert full_output.shape == x.shape
        assert lightweight_output.shape == x.shape
