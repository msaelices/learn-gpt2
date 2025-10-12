"""Tests for Problem 4: Multi-Head Attention."""

import pytest
import torch
import torch.nn as nn
from solution import MultiHeadAttention


def test_initialization():
    """Test that MultiHeadAttention initializes without errors."""
    mha = MultiHeadAttention(n_embd=768, n_head=12, dropout=0.1)
    assert mha is not None
    assert mha.n_embd == 768
    assert mha.n_head == 12
    assert mha.head_dim == 64


def test_invalid_head_configuration():
    """Test that initialization fails when n_embd is not divisible by n_head."""
    with pytest.raises(AssertionError):
        MultiHeadAttention(n_embd=768, n_head=11)  # 768 not divisible by 11


def test_forward_shape():
    """Test that forward pass produces correct output shape."""
    batch_size, seq_len, n_embd = 2, 10, 768
    n_head = 12

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.1)
    x = torch.randn(batch_size, seq_len, n_embd)
    output = mha(x)

    # Output shape should match input shape
    assert output.shape == (batch_size, seq_len, n_embd)


def test_single_head_attention():
    """Test multi-head attention with n_head=1 (should work like single-head)."""
    batch_size, seq_len, n_embd = 2, 8, 64
    n_head = 1

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    mha.eval()  # Disable dropout for deterministic testing

    x = torch.randn(batch_size, seq_len, n_embd)
    output = mha(x)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_multiple_heads():
    """Test with different numbers of heads."""
    batch_size, seq_len, n_embd = 2, 10, 768

    for n_head in [1, 2, 4, 6, 12]:
        mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.1)
        x = torch.randn(batch_size, seq_len, n_embd)
        output = mha(x)
        assert output.shape == (batch_size, seq_len, n_embd)


def test_different_sequence_lengths():
    """Test that attention works with different sequence lengths."""
    batch_size, n_embd, n_head = 2, 768, 12
    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.1)

    for seq_len in [1, 5, 10, 50, 100]:
        x = torch.randn(batch_size, seq_len, n_embd)
        output = mha(x)
        assert output.shape == (batch_size, seq_len, n_embd)


def test_batch_independence():
    """Test that different batch elements are processed independently."""
    batch_size, seq_len, n_embd, n_head = 2, 5, 64, 4

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    mha.eval()

    # Create input where second batch element is different
    x = torch.randn(batch_size, seq_len, n_embd)
    x_single = x[0:1]  # First batch element only

    # Process full batch and single element
    output_batch = mha(x)
    output_single = mha(x_single)

    # First element of batch output should match single element output
    assert torch.allclose(output_batch[0], output_single[0], atol=1e-6)


def test_gradient_flow():
    """Test that gradients flow through multi-head attention."""
    batch_size, seq_len, n_embd, n_head = 2, 5, 64, 4

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.1)
    x = torch.randn(batch_size, seq_len, n_embd, requires_grad=True)

    output = mha(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_dropout_training_vs_eval():
    """Test that dropout behaves differently in train vs eval mode."""
    batch_size, seq_len, n_embd, n_head = 2, 10, 64, 4

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.5)
    x = torch.randn(batch_size, seq_len, n_embd)

    # Training mode - should have stochastic behavior
    mha.train()
    output1 = mha(x)
    output2 = mha(x)
    # Outputs should be different due to dropout (with high probability)
    assert not torch.allclose(output1, output2, atol=1e-5)

    # Eval mode - should be deterministic
    mha.eval()
    output3 = mha(x)
    output4 = mha(x)
    assert torch.allclose(output3, output4, atol=1e-6)


def test_attention_layer_components():
    """Test that attention has the expected layer components."""
    mha = MultiHeadAttention(n_embd=768, n_head=12, dropout=0.1)

    # Check that required layers exist
    assert hasattr(mha, "c_attn")
    assert hasattr(mha, "c_proj")
    assert hasattr(mha, "attn_dropout")
    assert hasattr(mha, "resid_dropout")

    # Check layer types
    assert isinstance(mha.c_attn, nn.Linear)
    assert isinstance(mha.c_proj, nn.Linear)
    assert isinstance(mha.attn_dropout, nn.Dropout)
    assert isinstance(mha.resid_dropout, nn.Dropout)

    # Check layer dimensions
    assert mha.c_attn.in_features == 768
    assert mha.c_attn.out_features == 768 * 3  # Combined Q, K, V
    assert mha.c_proj.in_features == 768
    assert mha.c_proj.out_features == 768


def test_head_dimension_calculation():
    """Test that head dimension is calculated correctly."""
    test_cases = [
        (768, 12, 64),  # GPT-2 small
        (1024, 16, 64),  # GPT-2 medium
        (1280, 20, 64),  # GPT-2 large
        (1600, 25, 64),  # GPT-2 xl
    ]

    for n_embd, n_head, expected_head_dim in test_cases:
        mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head)
        assert mha.head_dim == expected_head_dim


def test_attention_with_zero_input():
    """Test behavior with zero input."""
    batch_size, seq_len, n_embd, n_head = 2, 5, 64, 4

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    mha.eval()

    x = torch.zeros(batch_size, seq_len, n_embd)
    output = mha(x)

    # Output should have correct shape
    assert output.shape == (batch_size, seq_len, n_embd)
    # Should not produce NaN or Inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_large_embedding_dimension():
    """Test attention with large embedding dimensions (numerical stability)."""
    batch_size, seq_len, n_embd, n_head = 1, 5, 2048, 16

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    mha.eval()

    x = torch.randn(batch_size, seq_len, n_embd)
    output = mha(x)

    # Check for numerical stability
    assert output.shape == (batch_size, seq_len, n_embd)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_output_changes_with_input():
    """Test that different inputs produce different outputs."""
    batch_size, seq_len, n_embd, n_head = 2, 5, 64, 4

    mha = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    mha.eval()

    x1 = torch.randn(batch_size, seq_len, n_embd)
    x2 = torch.randn(batch_size, seq_len, n_embd)

    output1 = mha(x1)
    output2 = mha(x2)

    # Different inputs should produce different outputs
    assert not torch.allclose(output1, output2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
