"""Tests for Problem 7: Layer Normalization & Residuals."""

import pytest
import torch
import torch.nn as nn
from solution import ResidualConnection


def test_residual_initialization():
    """Test that ResidualConnection initializes without errors."""
    res = ResidualConnection(n_embd=768, dropout=0.1)
    assert res is not None
    assert hasattr(res, 'ln')
    assert hasattr(res, 'dropout')


def test_residual_layer_norm_exists():
    """Test that layer normalization is properly initialized."""
    res = ResidualConnection(n_embd=768)
    assert isinstance(res.ln, nn.LayerNorm)
    assert res.ln.normalized_shape == (768,)


def test_residual_dropout_exists():
    """Test that dropout is properly initialized."""
    res = ResidualConnection(n_embd=768, dropout=0.5)
    assert isinstance(res.dropout, nn.Dropout)
    assert res.dropout.p == 0.5


def test_residual_forward_shape():
    """Test that ResidualConnection preserves shape."""
    res = ResidualConnection(n_embd=768, dropout=0.0)
    res.eval()

    # Identity sublayer (just returns input)
    identity = lambda x: x

    batch_size, seq_len, n_embd = 2, 10, 768
    x = torch.randn(batch_size, seq_len, n_embd)
    output = res(x, identity)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_residual_connection_applied():
    """Test that residual connection is actually applied."""
    res = ResidualConnection(n_embd=768, dropout=0.0)
    res.eval()

    # Zero sublayer (returns zeros)
    zero_sublayer = lambda x: torch.zeros_like(x)

    x = torch.randn(2, 10, 768)
    output = res(x, zero_sublayer)

    # With zero sublayer and residual, output should equal input
    assert torch.allclose(output, x, atol=1e-6)


def test_layer_norm_normalizes():
    """Test that layer normalization normalizes the last dimension."""
    res = ResidualConnection(n_embd=768, dropout=0.0)
    res.eval()

    # Identity sublayer
    identity = lambda x: torch.zeros_like(x)  # Zero to isolate layer norm effect

    x = torch.randn(2, 10, 768) * 100  # Large values

    # Apply just the layer norm (via forward with zero sublayer)
    with torch.no_grad():
        normalized = res.ln(x)

    # Check that mean ≈ 0 and std ≈ 1 across last dimension
    mean = normalized.mean(dim=-1)
    std = normalized.std(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


def test_residual_with_linear_sublayer():
    """Test residual connection with a simple linear sublayer."""
    res = ResidualConnection(n_embd=64, dropout=0.0)
    res.eval()

    # Simple linear transformation as sublayer
    linear = nn.Linear(64, 64)
    sublayer = lambda x: linear(x)

    x = torch.randn(2, 5, 64)
    output = res(x, sublayer)

    # Output should not equal input (sublayer transforms it)
    assert not torch.allclose(output, x, atol=1e-3)

    # But shape should be preserved
    assert output.shape == x.shape


def test_residual_gradient_flow():
    """Test that gradients flow through residual connection."""
    res = ResidualConnection(n_embd=768, dropout=0.0)

    # Simple sublayer with learnable parameters
    linear = nn.Linear(768, 768)
    sublayer = lambda x: linear(x)

    x = torch.randn(2, 10, 768, requires_grad=True)
    output = res(x, sublayer)
    loss = output.sum()
    loss.backward()

    # Gradients should flow to input
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    # Gradients should flow to sublayer parameters
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None


def test_residual_direct_gradient_path():
    """Test that residual provides direct gradient path."""
    res = ResidualConnection(n_embd=64, dropout=0.0)

    # Identity sublayer (minimal transformation)
    identity = lambda x: x * 0  # Zero output to test residual path

    x = torch.randn(2, 5, 64, requires_grad=True)
    output = res(x, identity)
    loss = output.sum()
    loss.backward()

    # With zero sublayer, gradient should still flow directly via residual
    assert x.grad is not None
    # Gradient should be approximately 1 (from the +1 in residual)
    assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-5)


def test_residual_dropout_training_vs_eval():
    """Test that dropout behaves differently in train vs eval mode."""
    res = ResidualConnection(n_embd=768, dropout=0.5)

    # Simple sublayer
    sublayer = lambda x: x * 2

    x = torch.randn(2, 10, 768)

    # Training mode - outputs should differ
    res.train()
    output1 = res(x, sublayer)
    output2 = res(x, sublayer)
    assert not torch.allclose(output1, output2, atol=1e-5)

    # Eval mode - outputs should be identical
    res.eval()
    output3 = res(x, sublayer)
    output4 = res(x, sublayer)
    assert torch.allclose(output3, output4, atol=1e-6)


def test_residual_no_dropout():
    """Test ResidualConnection with dropout=0."""
    res = ResidualConnection(n_embd=768, dropout=0.0)
    res.train()  # Even in train mode

    sublayer = lambda x: x * 2

    x = torch.randn(2, 10, 768)
    output1 = res(x, sublayer)
    output2 = res(x, sublayer)

    # Should be identical even in train mode
    assert torch.allclose(output1, output2, atol=1e-6)


def test_residual_epsilon_parameter():
    """Test that layer_norm_epsilon is properly used."""
    epsilon = 1e-6
    res = ResidualConnection(n_embd=768, layer_norm_epsilon=epsilon)
    assert res.ln.eps == epsilon


def test_residual_different_dimensions():
    """Test ResidualConnection with different embedding dimensions."""
    dimensions = [64, 256, 768, 1024, 2048]

    for n_embd in dimensions:
        res = ResidualConnection(n_embd=n_embd, dropout=0.0)
        res.eval()

        sublayer = lambda x: x * 0.5

        x = torch.randn(2, 5, n_embd)
        output = res(x, sublayer)

        assert output.shape == (2, 5, n_embd)


def test_residual_numerical_stability():
    """Test ResidualConnection with large values."""
    res = ResidualConnection(n_embd=768, dropout=0.0)
    res.eval()

    # Large input values
    x = torch.randn(2, 10, 768) * 1000

    sublayer = lambda x: x * 0.1

    output = res(x, sublayer)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_residual_pre_norm_order():
    """Test that pre-norm applies layer norm BEFORE sublayer."""
    res = ResidualConnection(n_embd=64, dropout=0.0)
    res.eval()

    # Track what the sublayer receives
    received_input = None

    def tracking_sublayer(x):
        nonlocal received_input
        received_input = x.clone()
        return x * 0

    x = torch.randn(1, 5, 64) * 100  # Large values

    with torch.no_grad():
        output = res(x, tracking_sublayer)
        expected_normalized = res.ln(x)

    # Sublayer should receive normalized input, not original
    assert received_input is not None
    assert torch.allclose(received_input, expected_normalized, atol=1e-5)

    # Sublayer should NOT receive original input
    assert not torch.allclose(received_input, x, atol=1.0)


def test_residual_with_multiple_calls():
    """Test that ResidualConnection can be called multiple times."""
    res = ResidualConnection(n_embd=768, dropout=0.0)
    res.eval()

    sublayer1 = lambda x: x * 0.5
    sublayer2 = lambda x: x * 2.0

    x = torch.randn(2, 10, 768)

    # Apply with different sublayers
    output1 = res(x, sublayer1)
    output2 = res(output1, sublayer2)

    assert output1.shape == x.shape
    assert output2.shape == x.shape
    assert not torch.allclose(output1, output2, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
