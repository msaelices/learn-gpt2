"""Problem 4: Multi-Head Attention - Solution

This solution implements multi-head self-attention, extracted from the reference
implementation in src/gpt2/model.py.
"""

import torch
import torch.nn as nn
from torch import Tensor


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
