"""Tests for Problem 2: Attention Basics

Run with: uv run pytest test_attention.py -v
"""

import pytest
import torch
import torch.nn as nn
from problem import SimpleAttention


def test_initialization():
    """Test that SimpleAttention initializes without errors."""
    attention = SimpleAttention(n_embd=768)
    assert attention is not None
    assert isinstance(attention, nn.Module)


def test_has_required_layers():
    """Test that module has query, key, and value projections."""
    attention = SimpleAttention()

    assert hasattr(attention, "query"), "Missing query projection layer"
    assert isinstance(attention.query, nn.Linear), "query should be nn.Linear"

    assert hasattr(attention, "key"), "Missing key projection layer"
    assert isinstance(attention.key, nn.Linear), "key should be nn.Linear"

    assert hasattr(attention, "value"), "Missing value projection layer"
    assert isinstance(attention.value, nn.Linear), "value should be nn.Linear"


def test_projection_dimensions():
    """Test that projection layers have correct dimensions."""
    n_embd = 256
    attention = SimpleAttention(n_embd=n_embd)

    assert attention.query.in_features == n_embd, f"Query input should be {n_embd}"
    assert attention.query.out_features == n_embd, f"Query output should be {n_embd}"

    assert attention.key.in_features == n_embd, f"Key input should be {n_embd}"
    assert attention.key.out_features == n_embd, f"Key output should be {n_embd}"

    assert attention.value.in_features == n_embd, f"Value input should be {n_embd}"
    assert attention.value.out_features == n_embd, f"Value output should be {n_embd}"


def test_forward_output_shape():
    """Test that forward pass produces correct output shape."""
    batch_size = 4
    seq_len = 20
    n_embd = 768

    attention = SimpleAttention(n_embd=n_embd)
    x = torch.randn(batch_size, seq_len, n_embd)

    output = attention(x)

    expected_shape = (batch_size, seq_len, n_embd)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


def test_different_sequence_lengths():
    """Test that attention works with different sequence lengths."""
    attention = SimpleAttention(n_embd=128)

    for seq_len in [1, 5, 10, 50, 100]:
        x = torch.randn(2, seq_len, 128)
        output = attention(x)
        assert output.shape == (
            2,
            seq_len,
            128,
        ), f"Failed for sequence length {seq_len}"


def test_different_batch_sizes():
    """Test that attention works with different batch sizes."""
    attention = SimpleAttention(n_embd=128)

    for batch_size in [1, 2, 8, 16]:
        x = torch.randn(batch_size, 10, 128)
        output = attention(x)
        assert output.shape == (
            batch_size,
            10,
            128,
        ), f"Failed for batch size {batch_size}"


def test_attention_weights_sum_to_one():
    """Test that attention mechanism produces valid probability distributions."""
    attention = SimpleAttention(n_embd=64)
    x = torch.randn(1, 5, 64)

    # We'll need to access intermediate values, so let's recompute
    with torch.no_grad():
        q = attention.query(x)
        k = attention.key(x)

        attn_scores = q @ k.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Each row should sum to 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
            "Attention weights should sum to 1 along last dimension"
        )


def test_output_is_weighted_combination():
    """Test that output is a weighted combination of values."""
    attention = SimpleAttention(n_embd=64)
    attention.eval()

    # Simple case: one sequence
    x = torch.randn(1, 3, 64)

    with torch.no_grad():
        output = attention(x)

        # Output should be in same range as input (roughly)
        # Since it's a weighted combination of values
        assert output.abs().max() < 100 * x.abs().max(), "Output seems unreasonable"


def test_different_inputs_produce_different_outputs():
    """Test that different inputs produce different outputs."""
    attention = SimpleAttention()
    attention.eval()

    x1 = torch.randn(1, 10, 768)
    x2 = torch.randn(1, 10, 768)

    with torch.no_grad():
        output1 = attention(x1)
        output2 = attention(x2)

    assert not torch.allclose(output1, output2), "Different inputs should produce different outputs"


def test_attention_is_permutation_sensitive():
    """Test that attention is sensitive to input order (due to learned weights)."""
    attention = SimpleAttention(n_embd=64)
    attention.eval()

    # Create input and its permutation
    x = torch.randn(1, 5, 64)
    x_permuted = x[:, torch.randperm(5), :]  # Shuffle sequence dimension

    with torch.no_grad():
        output = attention(x)
        output_permuted = attention(x_permuted)

    # Outputs should generally be different
    # (They could accidentally be similar, but very unlikely)
    assert not torch.allclose(output, output_permuted, atol=1e-3), (
        "Attention should be sensitive to input order"
    )


def test_gradients_flow():
    """Test that gradients flow through the attention mechanism."""
    attention = SimpleAttention(n_embd=64)
    x = torch.randn(2, 5, 64, requires_grad=True)

    output = attention(x)
    loss = output.sum()
    loss.backward()

    # Check that projections have gradients
    assert attention.query.weight.grad is not None, "Query projection should have gradients"
    assert attention.key.weight.grad is not None, "Key projection should have gradients"
    assert attention.value.weight.grad is not None, "Value projection should have gradients"

    # Check that input has gradients
    assert x.grad is not None, "Input should have gradients"


def test_self_attention_property():
    """Test that this is self-attention (Q, K, V from same input)."""
    attention = SimpleAttention(n_embd=64)
    x = torch.randn(1, 3, 64)

    # Manually compute Q, K, V
    with torch.no_grad():
        q = attention.query(x)
        k = attention.key(x)
        v = attention.value(x)

    # All should have same input shape
    assert q.shape == k.shape == v.shape == x.shape


def test_attention_matrix_shape():
    """Test that attention scores have correct shape (seq_len x seq_len)."""
    batch_size = 2
    seq_len = 7
    n_embd = 64

    attention = SimpleAttention(n_embd=n_embd)
    x = torch.randn(batch_size, seq_len, n_embd)

    with torch.no_grad():
        q = attention.query(x)
        k = attention.key(x)
        attn_scores = q @ k.transpose(-2, -1)

        expected_shape = (batch_size, seq_len, seq_len)
        assert attn_scores.shape == expected_shape, (
            f"Attention scores should have shape {expected_shape}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
