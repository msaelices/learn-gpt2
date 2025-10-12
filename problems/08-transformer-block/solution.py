"""Problem 8: Complete Transformer Block - Solution

This solution implements the full transformer block with pre-norm architecture.
It combines causal multi-head attention and feedforward network with residual connections.
"""

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    This is extracted from the reference implementation for use in TransformerBlock.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_positions: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ) -> None:
        """Initialize multi-head attention.

        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            n_positions: Maximum sequence length.
            attn_pdrop: Attention dropout probability.
            resid_pdrop: Residual dropout probability.
        """
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Combined Q, K, V projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropout layers
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_positions, n_positions)).view(
                1, 1, n_positions, n_positions
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        batch_size, seq_len, n_embd = x.size()

        # Calculate Q, K, V for all heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to split into multiple heads
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v

        # Concatenate heads and reshape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        # Output projection and dropout
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


class NewGELU(nn.Module):
    """GELU activation function (GPT-2 approximation)."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation."""
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, n_embd: int, n_inner: int, dropout: float = 0.1) -> None:
        """Initialize feedforward network.

        Args:
            n_embd: Embedding dimension.
            n_inner: Inner/hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.c_fc = nn.Linear(n_embd, n_inner)
        self.c_proj = nn.Linear(n_inner, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.act = NewGELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Complete transformer block: Attention + FFN with residuals.

    Implements pre-norm architecture:
        1. x = x + attention(layer_norm_1(x))
        2. x = x + feedforward(layer_norm_2(x))

    This is the fundamental building block of GPT-2.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_inner: int = None,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        """Initialize transformer block.

        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            n_positions: Maximum sequence length.
            n_inner: Inner dimension of feedforward network. Defaults to 4 * n_embd.
            attn_pdrop: Attention dropout probability.
            resid_pdrop: Residual dropout probability.
            layer_norm_epsilon: Epsilon for layer normalization.
        """
        super().__init__()
        if n_inner is None:
            n_inner = 4 * n_embd

        # Layer normalization before attention (pre-norm)
        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

        # Multi-head self-attention with causal masking
        self.attn = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_positions=n_positions,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        # Layer normalization before feedforward (pre-norm)
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

        # Feedforward network (MLP)
        self.mlp = FeedForward(
            n_embd=n_embd,
            n_inner=n_inner,
            dropout=resid_pdrop,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        The transformation flow:
            1. Normalize, apply attention, add residual
            2. Normalize, apply feedforward, add residual
        """
        # Attention block with residual connection (pre-norm)
        # x = x + attention(layer_norm(x))
        x = x + self.attn(self.ln_1(x))

        # Feedforward block with residual connection (pre-norm)
        # x = x + feedforward(layer_norm(x))
        x = x + self.mlp(self.ln_2(x))

        return x
