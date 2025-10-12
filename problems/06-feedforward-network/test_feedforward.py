"""Tests for Problem 6: Feedforward Network."""

import pytest
import torch
import torch.nn as nn
from solution import FeedForward, NewGELU


def test_gelu_initialization():
    """Test that NewGELU initializes without errors."""
    gelu = NewGELU()
    assert gelu is not None


def test_gelu_forward_shape():
    """Test that GELU preserves input shape."""
    gelu = NewGELU()

    # Test various shapes
    shapes = [(10,), (5, 10), (2, 5, 10), (2, 5, 10, 768)]
    for shape in shapes:
        x = torch.randn(shape)
        output = gelu(x)
        assert output.shape == x.shape


def test_gelu_zero():
    """Test GELU at zero (should be close to 0)."""
    gelu = NewGELU()
    x = torch.tensor([0.0])
    output = gelu(x)
    assert torch.isclose(output, torch.tensor(0.0), atol=1e-6)


def test_gelu_known_values():
    """Test GELU with known values."""
    gelu = NewGELU()

    # GELU(1.0) ≈ 0.841
    x = torch.tensor([1.0])
    output = gelu(x)
    assert torch.isclose(output, torch.tensor(0.841), atol=0.01)

    # GELU(-1.0) ≈ -0.159
    x = torch.tensor([-1.0])
    output = gelu(x)
    assert torch.isclose(output, torch.tensor(-0.159), atol=0.01)


def test_gelu_comparison_with_pytorch():
    """Test that our GELU is close to PyTorch's GELU."""
    our_gelu = NewGELU()
    pytorch_gelu = nn.GELU(approximate='tanh')  # PyTorch's tanh approximation

    x = torch.randn(100)

    our_output = our_gelu(x)
    pytorch_output = pytorch_gelu(x)

    # Should be very close
    assert torch.allclose(our_output, pytorch_output, atol=1e-5)


def test_gelu_non_monotonic():
    """Test that GELU can output negative values (non-monotonic)."""
    gelu = NewGELU()
    x = torch.tensor([-1.0, -0.5, -0.1])
    output = gelu(x)

    # All should be negative (for moderate negative inputs)
    assert (output < 0).all()


def test_feedforward_initialization():
    """Test that FeedForward initializes without errors."""
    ff = FeedForward(n_embd=768, n_inner=3072, dropout=0.1)
    assert ff is not None
    assert hasattr(ff, 'c_fc')
    assert hasattr(ff, 'c_proj')
    assert hasattr(ff, 'act')
    assert hasattr(ff, 'dropout')


def test_feedforward_layer_dimensions():
    """Test that FeedForward layers have correct dimensions."""
    n_embd, n_inner = 768, 3072
    ff = FeedForward(n_embd=n_embd, n_inner=n_inner)

    # Check layer dimensions
    assert isinstance(ff.c_fc, nn.Linear)
    assert isinstance(ff.c_proj, nn.Linear)
    assert isinstance(ff.act, NewGELU)
    assert isinstance(ff.dropout, nn.Dropout)

    assert ff.c_fc.in_features == n_embd
    assert ff.c_fc.out_features == n_inner
    assert ff.c_proj.in_features == n_inner
    assert ff.c_proj.out_features == n_embd


def test_feedforward_forward_shape():
    """Test that FeedForward preserves shape."""
    batch_size, seq_len, n_embd = 2, 10, 768
    n_inner = 3072

    ff = FeedForward(n_embd=n_embd, n_inner=n_inner)
    x = torch.randn(batch_size, seq_len, n_embd)
    output = ff(x)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_feedforward_different_sizes():
    """Test FeedForward with different embedding dimensions."""
    configs = [
        (64, 256),    # Small
        (768, 3072),  # GPT-2 small
        (1024, 4096), # GPT-2 medium
        (1280, 5120), # GPT-2 large
    ]

    for n_embd, n_inner in configs:
        ff = FeedForward(n_embd=n_embd, n_inner=n_inner)
        x = torch.randn(2, 5, n_embd)
        output = ff(x)
        assert output.shape == (2, 5, n_embd)


def test_feedforward_gradient_flow():
    """Test that gradients flow through FeedForward."""
    ff = FeedForward(n_embd=768, n_inner=3072)
    x = torch.randn(2, 10, 768, requires_grad=True)

    output = ff(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_feedforward_dropout_training_vs_eval():
    """Test that dropout behaves differently in train vs eval mode."""
    ff = FeedForward(n_embd=768, n_inner=3072, dropout=0.5)
    x = torch.randn(2, 10, 768)

    # Training mode - outputs should differ
    ff.train()
    output1 = ff(x)
    output2 = ff(x)
    assert not torch.allclose(output1, output2, atol=1e-5)

    # Eval mode - outputs should be identical
    ff.eval()
    output3 = ff(x)
    output4 = ff(x)
    assert torch.allclose(output3, output4, atol=1e-6)


def test_feedforward_no_dropout():
    """Test FeedForward with dropout=0."""
    ff = FeedForward(n_embd=768, n_inner=3072, dropout=0.0)
    ff.train()  # Even in train mode

    x = torch.randn(2, 10, 768)
    output1 = ff(x)
    output2 = ff(x)

    # Should be identical even in train mode
    assert torch.allclose(output1, output2, atol=1e-6)


def test_feedforward_expansion_factor():
    """Test that common expansion factor is 4x."""
    n_embd = 768
    n_inner = 4 * n_embd  # Common 4x expansion

    ff = FeedForward(n_embd=n_embd, n_inner=n_inner)

    assert ff.c_fc.out_features == 4 * n_embd
    assert ff.c_proj.in_features == 4 * n_embd


def test_feedforward_changes_values():
    """Test that FeedForward actually transforms the input."""
    ff = FeedForward(n_embd=768, n_inner=3072, dropout=0.0)
    ff.eval()

    x = torch.randn(2, 10, 768)
    output = ff(x)

    # Output should be different from input
    assert not torch.allclose(x, output, atol=1e-3)


def test_feedforward_numerical_stability():
    """Test FeedForward with large dimensions."""
    ff = FeedForward(n_embd=2048, n_inner=8192, dropout=0.0)
    ff.eval()

    x = torch.randn(2, 20, 2048)
    output = ff(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_gelu_gradient():
    """Test that GELU has non-zero gradients."""
    gelu = NewGELU()
    x = torch.randn(100, requires_grad=True)

    output = gelu(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert (x.grad != 0).any()  # Should have non-zero gradients


def test_feedforward_position_wise():
    """Test that FeedForward is position-wise (independent per position)."""
    ff = FeedForward(n_embd=64, n_inner=256, dropout=0.0)
    ff.eval()

    batch_size, seq_len, n_embd = 1, 5, 64

    # Create input where each position is different
    x = torch.randn(batch_size, seq_len, n_embd)

    output = ff(x)

    # Process each position independently
    outputs_separate = []
    for i in range(seq_len):
        pos_input = x[:, i:i+1, :]  # Single position
        pos_output = ff(pos_input)
        outputs_separate.append(pos_output)

    outputs_separate = torch.cat(outputs_separate, dim=1)

    # Should be the same (position-wise operation)
    assert torch.allclose(output, outputs_separate, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
