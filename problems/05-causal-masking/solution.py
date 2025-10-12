"""Problem 5: Causal Masking - Solution

This solution implements multi-head self-attention with causal masking,
extracted from the reference implementation in src/gpt2/model.py.
"""

import torch
import torch.nn as nn
from torch import Tensor


class CausalMultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    This prevents positions from attending to subsequent positions,
    which is essential for autoregressive language modeling (like GPT-2).
    During generation, we predict one token at a time, so each position
    should only have access to previous positions.
    """

    def __init__(
        self, n_embd: int, n_head: int, n_positions: int = 1024, dropout: float = 0.1
    ) -> None:
        """Initialize causal multi-head attention.

        Args:
            n_embd: Embedding dimension (must be divisible by n_head).
            n_head: Number of attention heads.
            n_positions: Maximum sequence length for causal mask.
            dropout: Dropout probability.
        """
        super().__init__()
        # Validate configuration
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_positions = n_positions

        # Combined Q, K, V projection (more efficient than separate projections)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask to ensure attention only flows to the left
        # This is a lower-triangular matrix where mask[i,j] = 1 if j <= i
        # Shape: (1, 1, n_positions, n_positions) for broadcasting
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        batch_size, seq_len, n_embd = x.size()

        # Step 1: Project to Q, K, V all at once
        # Shape: (batch_size, seq_len, 3 * n_embd)
        qkv = self.c_attn(x)

        # Step 2: Split into Q, K, V
        # Each has shape: (batch_size, seq_len, n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Step 3: Reshape to split into multiple heads
        # From: (batch_size, seq_len, n_embd)
        # To: (batch_size, n_head, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Step 4: Compute scaled dot-product attention scores
        # (batch_size, n_head, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Step 5: Apply causal mask (prevent attending to future positions)
        # Slice the mask to match current sequence length
        # Set masked positions (where bias == 0) to -inf
        # After softmax, -inf becomes 0, effectively blocking attention
        attn_scores = attn_scores.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        # Step 6: Apply softmax to get attention weights
        # Masked positions (-inf) become 0 after softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Step 7: Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Step 8: Apply attention to values
        # (batch_size, n_head, seq_len, head_dim)
        out = attn_weights @ v

        # Step 9: Concatenate heads back together
        # Transpose to: (batch_size, seq_len, n_head, head_dim)
        # Then reshape to: (batch_size, seq_len, n_embd)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        # Step 10: Apply output projection and dropout
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


# Compare this snippet from problems/04-multi-head-attention/solution.py:º
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions. Instead of
    performing a single attention function with d_model-dimensional keys, values
    and queries, we linearly project the queries, keys and values h times with
    different, learned linear projections.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1) -> None:
        """Initialize multi-head attention.

        Args:
            n_embd: Embedding dimension (must be divisible by n_head).
            n_head: Number of attention heads.
            dropout: Dropout probability for attention weights and output.
        """
        super().__init__()
        # Validate that n_embd is divisible by n_head
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Combined Q, K, V projection (more efficient than separate projections)
        # Projects from n_embd to 3 * n_embd (for Q, K, V simultaneously)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

        # Output projection - projects concatenated heads back to n_embd
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)  # For attention weights
        self.resid_dropout = nn.Dropout(dropout)  # For final output

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        batch_size, seq_len, n_embd = x.size()

        # Step 1: Project input to Q, K, V all at once
        # Shape: (batch_size, seq_len, 3 * n_embd)
        qkv = self.c_attn(x)

        # Step 2: Split into Q, K, V
        # Each has shape: (batch_size, seq_len, n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Step 3: Reshape to split into multiple heads
        # From: (batch_size, seq_len, n_embd)
        # To: (batch_size, seq_len, n_head, head_dim)
        # Then transpose to: (batch_size, n_head, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Step 4: Compute scaled dot-product attention scores
        # (batch_size, n_head, seq_len, seq_len)
        # Note: We scale by head_dim, not n_embd!
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Step 5: Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Step 6: Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Step 7: Apply attention to values
        # (batch_size, n_head, seq_len, head_dim)
        out = attn_weights @ v

        # Step 8: Concatenate heads back together
        # Transpose back: (batch_size, seq_len, n_head, head_dim)
        # Then reshape to: (batch_size, seq_len, n_embd)
        # .contiguous() is needed because transpose() doesn't change memory layout
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        # Step 9: Apply output projection
        out = self.c_proj(out)

        # Step 10: Apply output dropout
        out = self.resid_dropout(out)

        return out
