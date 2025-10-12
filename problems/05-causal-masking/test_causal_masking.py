"""Tests for Problem 5: Causal Masking."""

import pytest
import torch
from solution import CausalMultiHeadAttention


def test_initialization():
    """Test that CausalMultiHeadAttention initializes without errors."""
    attn = CausalMultiHeadAttention(n_embd=768, n_head=12, n_positions=1024, dropout=0.1)
    assert attn is not None
    assert attn.n_embd == 768
    assert attn.n_head == 12
    assert attn.head_dim == 64
    assert attn.n_positions == 1024


def test_causal_mask_registered():
    """Test that causal mask is registered as a buffer."""
    attn = CausalMultiHeadAttention(n_embd=768, n_head=12, n_positions=1024)

    # Check that bias buffer exists
    assert hasattr(attn, "bias")
    assert "bias" in dict(attn.named_buffers())

    # Check mask shape
    assert attn.bias.shape == (1, 1, 1024, 1024)


def test_causal_mask_is_lower_triangular():
    """Test that the causal mask is lower-triangular."""
    attn = CausalMultiHeadAttention(n_embd=64, n_head=4, n_positions=10)

    mask = attn.bias.squeeze()

    # Check lower triangle (should be 1)
    for i in range(10):
        for j in range(i + 1):
            assert mask[i, j] == 1, f"mask[{i},{j}] should be 1 (lower triangle)"

    # Check upper triangle (should be 0)
    for i in range(10):
        for j in range(i + 1, 10):
            assert mask[i, j] == 0, f"mask[{i},{j}] should be 0 (upper triangle)"


def test_forward_shape():
    """Test that forward pass produces correct output shape."""
    batch_size, seq_len, n_embd = 2, 10, 768
    n_head = 12

    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.1)
    x = torch.randn(batch_size, seq_len, n_embd)
    output = attn(x)

    assert output.shape == (batch_size, seq_len, n_embd)


def test_causal_attention_pattern():
    """Test that attention is indeed causal (no future information)."""
    # Use small dimensions for easy verification
    batch_size, seq_len, n_embd, n_head = 1, 5, 64, 4

    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, n_positions=10, dropout=0.0)
    attn.eval()

    # Modified version to capture attention weights
    class CausalAttnWithWeights(CausalMultiHeadAttention):
        def forward(self, x):
            batch_size, seq_len, n_embd = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

            attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            attn_scores = attn_scores.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
            attn_weights = torch.softmax(attn_scores, dim=-1)

            out = attn_weights @ v
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
            out = self.c_proj(out)

            return out, attn_weights

    attn_with_weights = CausalAttnWithWeights(
        n_embd=n_embd, n_head=n_head, n_positions=10, dropout=0.0
    )
    attn_with_weights.load_state_dict(attn.state_dict())
    attn_with_weights.eval()

    x = torch.randn(batch_size, seq_len, n_embd)

    with torch.no_grad():
        _, attn_weights = attn_with_weights(x)

    # attn_weights shape: (batch, n_head, seq_len, seq_len)
    # Check that future positions have 0 attention
    for head in range(n_head):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Position i should not attend to position j when j > i
                assert attn_weights[0, head, i, j] == 0, (
                    f"Head {head}: Position {i} should not attend to future position {j}"
                )


def test_attention_weights_sum_to_one():
    """Test that attention weights for valid positions sum to 1."""
    batch_size, seq_len, n_embd, n_head = 1, 8, 64, 4

    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, n_positions=10, dropout=0.0)
    attn.eval()

    # Capture attention weights
    class CausalAttnWithWeights(CausalMultiHeadAttention):
        def forward(self, x):
            batch_size, seq_len, n_embd = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

            attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            attn_scores = attn_scores.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
            attn_weights = torch.softmax(attn_scores, dim=-1)

            out = attn_weights @ v
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
            out = self.c_proj(out)

            return out, attn_weights

    attn_with_weights = CausalAttnWithWeights(
        n_embd=n_embd, n_head=n_head, n_positions=10, dropout=0.0
    )
    attn_with_weights.load_state_dict(attn.state_dict())
    attn_with_weights.eval()

    x = torch.randn(batch_size, seq_len, n_embd)

    with torch.no_grad():
        _, attn_weights = attn_with_weights(x)

    # Check that each position's attention weights sum to 1
    # attn_weights shape: (batch, n_head, seq_len, seq_len)
    row_sums = attn_weights.sum(dim=-1)  # Sum over key dimension

    # All sums should be 1.0
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_different_sequence_lengths():
    """Test that masking works with different sequence lengths."""
    batch_size, n_embd, n_head = 2, 768, 12
    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, n_positions=1024)

    for seq_len in [1, 5, 10, 50, 100, 500]:
        x = torch.randn(batch_size, seq_len, n_embd)
        output = attn(x)
        assert output.shape == (batch_size, seq_len, n_embd)


def test_mask_device_consistency():
    """Test that mask moves with model to correct device."""
    attn = CausalMultiHeadAttention(n_embd=768, n_head=12)

    # Check initial device
    assert attn.bias.device.type == "cpu"

    # Move to CPU explicitly (should work)
    attn.to("cpu")
    assert attn.bias.device.type == "cpu"

    # If CUDA available, test GPU
    if torch.cuda.is_available():
        attn.to("cuda")
        assert attn.bias.device.type == "cuda"


def test_gradient_flow():
    """Test that gradients flow through causal attention."""
    batch_size, seq_len, n_embd, n_head = 2, 5, 64, 4

    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.1)
    x = torch.randn(batch_size, seq_len, n_embd, requires_grad=True)

    output = attn(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_first_position_only_self_attention():
    """Test that first position can only attend to itself."""
    batch_size, seq_len, n_embd, n_head = 1, 5, 64, 4

    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    attn.eval()

    class CausalAttnWithWeights(CausalMultiHeadAttention):
        def forward(self, x):
            batch_size, seq_len, n_embd = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

            attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            attn_scores = attn_scores.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
            attn_weights = torch.softmax(attn_scores, dim=-1)

            out = attn_weights @ v
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
            out = self.c_proj(out)

            return out, attn_weights

    attn_with_weights = CausalAttnWithWeights(n_embd=n_embd, n_head=n_head, dropout=0.0)
    attn_with_weights.load_state_dict(attn.state_dict())
    attn_with_weights.eval()

    x = torch.randn(batch_size, seq_len, n_embd)

    with torch.no_grad():
        _, attn_weights = attn_with_weights(x)

    # First position (i=0) should have attention weight 1.0 at position 0
    # and 0.0 everywhere else
    for head in range(n_head):
        assert torch.isclose(attn_weights[0, head, 0, 0], torch.tensor(1.0), atol=1e-6)
        for j in range(1, seq_len):
            assert attn_weights[0, head, 0, j] == 0


def test_last_position_attends_to_all():
    """Test that last position can attend to all previous positions."""
    batch_size, seq_len, n_embd, n_head = 1, 5, 64, 4

    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
    attn.eval()

    class CausalAttnWithWeights(CausalMultiHeadAttention):
        def forward(self, x):
            batch_size, seq_len, n_embd = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

            attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            attn_scores = attn_scores.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
            attn_weights = torch.softmax(attn_scores, dim=-1)

            out = attn_weights @ v
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
            out = self.c_proj(out)

            return out, attn_weights

    attn_with_weights = CausalAttnWithWeights(n_embd=n_embd, n_head=n_head, dropout=0.0)
    attn_with_weights.load_state_dict(attn.state_dict())
    attn_with_weights.eval()

    x = torch.randn(batch_size, seq_len, n_embd)

    with torch.no_grad():
        _, attn_weights = attn_with_weights(x)

    # Last position (i=seq_len-1) should have non-zero attention to all positions
    # and sum to 1.0
    for head in range(n_head):
        last_pos_weights = attn_weights[0, head, seq_len - 1, :]
        assert (last_pos_weights > 0).all(), "Last position should attend to all positions"
        assert torch.isclose(last_pos_weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_dropout_behavior():
    """Test that dropout behaves differently in train vs eval mode."""
    batch_size, seq_len, n_embd, n_head = 2, 10, 64, 4

    attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.5)
    x = torch.randn(batch_size, seq_len, n_embd)

    # Training mode - outputs should differ
    attn.train()
    output1 = attn(x)
    output2 = attn(x)
    assert not torch.allclose(output1, output2, atol=1e-5)

    # Eval mode - outputs should be identical
    attn.eval()
    output3 = attn(x)
    output4 = attn(x)
    assert torch.allclose(output3, output4, atol=1e-6)


def test_numerical_stability():
    """Test numerical stability with large dimensions."""
    batch_size, seq_len = 2, 20

    for n_embd in [256, 512, 1024, 2048]:
        n_head = n_embd // 64  # Standard head dimension
        attn = CausalMultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=0.0)
        attn.eval()

        x = torch.randn(batch_size, seq_len, n_embd)

        with torch.no_grad():
            output = attn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
