"""Tests for Problem 8: Complete Transformer Block."""

import pytest
import torch
import torch.nn as nn
from solution import TransformerBlock


def test_transformer_block_initialization():
    """Test that TransformerBlock initializes without errors."""
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
    )
    assert block is not None
    assert hasattr(block, 'ln_1')
    assert hasattr(block, 'attn')
    assert hasattr(block, 'ln_2')
    assert hasattr(block, 'mlp')


def test_transformer_block_components():
    """Test that TransformerBlock has correct components."""
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
    )

    # Check layer norms
    assert isinstance(block.ln_1, nn.LayerNorm)
    assert isinstance(block.ln_2, nn.LayerNorm)
    assert block.ln_1.normalized_shape == (768,)
    assert block.ln_2.normalized_shape == (768,)

    # Check attention and MLP
    assert hasattr(block.attn, 'c_attn')  # MultiHeadAttention
    assert hasattr(block.mlp, 'c_fc')     # FeedForward


def test_transformer_block_forward_shape():
    """Test that TransformerBlock preserves shape."""
    batch_size, seq_len, n_embd = 2, 10, 768

    block = TransformerBlock(
        n_embd=n_embd,
        n_head=12,
        n_positions=1024,
    )
    block.eval()

    x = torch.randn(batch_size, seq_len, n_embd)
    output = block(x)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_transformer_block_different_configs():
    """Test TransformerBlock with different configurations."""
    configs = [
        (64, 4, 128),       # Small
        (768, 12, 1024),    # GPT-2 small
        (1024, 16, 1024),   # GPT-2 medium
        (1280, 20, 1024),   # GPT-2 large
    ]

    for n_embd, n_head, n_positions in configs:
        block = TransformerBlock(
            n_embd=n_embd,
            n_head=n_head,
            n_positions=n_positions,
        )
        block.eval()

        x = torch.randn(2, 10, n_embd)
        output = block(x)

        assert output.shape == (2, 10, n_embd)


def test_transformer_block_gradient_flow():
    """Test that gradients flow through the entire block."""
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
    )

    x = torch.randn(2, 10, 768, requires_grad=True)
    output = block(x)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    # Check layer norm gradients
    assert block.ln_1.weight.grad is not None
    assert block.ln_2.weight.grad is not None


def test_transformer_block_causal_masking():
    """Test that causal masking is applied (future tokens don't affect past)."""
    block = TransformerBlock(
        n_embd=64,
        n_head=4,
        n_positions=128,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    block.eval()

    seq_len = 5
    x = torch.randn(1, seq_len, 64)

    # Forward pass
    output = block(x)

    # Change a future token and check that past tokens are unchanged
    x_modified = x.clone()
    x_modified[0, -1, :] = torch.randn(64)  # Modify last token

    output_modified = block(x_modified)

    # First tokens should be very similar (within tolerance due to layer norm)
    # They should not be identical due to layer norm statistics changing
    # But the difference should be very small
    for i in range(seq_len - 1):
        diff = (output[0, i, :] - output_modified[0, i, :]).abs().mean()
        assert diff < 0.1, f"Position {i} affected too much by future change: {diff}"


def test_transformer_block_changes_input():
    """Test that TransformerBlock actually transforms the input."""
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    block.eval()

    x = torch.randn(2, 10, 768)
    output = block(x)

    # Output should be different from input
    assert not torch.allclose(x, output, atol=1e-3)


def test_transformer_block_residual_connections():
    """Test that residual connections are working."""
    # Create block with very small weights to test residual path
    block = TransformerBlock(
        n_embd=64,
        n_head=4,
        n_positions=128,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )

    # Initialize weights to very small values
    for param in block.parameters():
        param.data.fill_(0.001)

    block.eval()

    x = torch.randn(1, 5, 64)
    output = block(x)

    # With very small weights, output should be close to input due to residuals
    # (residual adds back the original input)
    assert torch.allclose(output, x, atol=0.5)


def test_transformer_block_dropout_training_vs_eval():
    """Test that dropout behaves differently in train vs eval mode."""
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        attn_pdrop=0.5,
        resid_pdrop=0.5,
    )

    x = torch.randn(2, 10, 768)

    # Training mode - outputs should differ
    block.train()
    output1 = block(x)
    output2 = block(x)
    assert not torch.allclose(output1, output2, atol=1e-5)

    # Eval mode - outputs should be identical
    block.eval()
    output3 = block(x)
    output4 = block(x)
    assert torch.allclose(output3, output4, atol=1e-6)


def test_transformer_block_sequence_length_handling():
    """Test TransformerBlock with variable sequence lengths."""
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
    )
    block.eval()

    seq_lengths = [1, 5, 10, 50, 100]

    for seq_len in seq_lengths:
        x = torch.randn(2, seq_len, 768)
        output = block(x)
        assert output.shape == (2, seq_len, 768)


def test_transformer_block_numerical_stability():
    """Test TransformerBlock with large values."""
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
    )
    block.eval()

    # Large input values
    x = torch.randn(2, 10, 768) * 1000

    output = block(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_transformer_block_inner_dimension():
    """Test TransformerBlock with custom inner dimension."""
    n_embd = 768

    # Test with default (4 * n_embd)
    block1 = TransformerBlock(
        n_embd=n_embd,
        n_head=12,
        n_positions=1024,
    )
    assert block1.mlp.c_fc.out_features == 4 * n_embd

    # Test with custom inner dimension
    n_inner = 2048
    block2 = TransformerBlock(
        n_embd=n_embd,
        n_head=12,
        n_positions=1024,
        n_inner=n_inner,
    )
    assert block2.mlp.c_fc.out_features == n_inner


def test_transformer_block_layer_norm_epsilon():
    """Test that layer_norm_epsilon is properly set."""
    epsilon = 1e-6
    block = TransformerBlock(
        n_embd=768,
        n_head=12,
        n_positions=1024,
        layer_norm_epsilon=epsilon,
    )

    assert block.ln_1.eps == epsilon
    assert block.ln_2.eps == epsilon


def test_transformer_block_attention_output():
    """Test that attention sublayer produces valid output."""
    block = TransformerBlock(
        n_embd=64,
        n_head=4,
        n_positions=128,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    block.eval()

    x = torch.randn(1, 5, 64)

    # Test attention sublayer independently
    with torch.no_grad():
        ln1_out = block.ln_1(x)
        attn_out = block.attn(ln1_out)

    assert attn_out.shape == x.shape
    assert not torch.isnan(attn_out).any()


def test_transformer_block_feedforward_output():
    """Test that feedforward sublayer produces valid output."""
    block = TransformerBlock(
        n_embd=64,
        n_head=4,
        n_positions=128,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    block.eval()

    x = torch.randn(1, 5, 64)

    # Test feedforward sublayer independently
    with torch.no_grad():
        # First apply attention block
        x_after_attn = x + block.attn(block.ln_1(x))
        # Then test feedforward
        ln2_out = block.ln_2(x_after_attn)
        mlp_out = block.mlp(ln2_out)

    assert mlp_out.shape == x.shape
    assert not torch.isnan(mlp_out).any()


def test_transformer_block_parameter_count():
    """Test that parameter count is reasonable."""
    n_embd = 768
    n_head = 12

    block = TransformerBlock(
        n_embd=n_embd,
        n_head=n_head,
        n_positions=1024,
    )

    total_params = sum(p.numel() for p in block.parameters())

    # Approximate expected parameters:
    # - Attention: 3 * n_embd * n_embd (QKV) + n_embd * n_embd (proj) + biases
    # - FFN: n_embd * 4*n_embd + 4*n_embd * n_embd + biases
    # - LayerNorm: 2 * (n_embd + n_embd) for weights and biases
    expected_approx = (3 * n_embd * n_embd) + (n_embd * n_embd) + \
                      (n_embd * 4 * n_embd) + (4 * n_embd * n_embd)

    # Should be within reasonable range (allow for biases and layer norms)
    assert total_params > expected_approx * 0.9
    assert total_params < expected_approx * 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
