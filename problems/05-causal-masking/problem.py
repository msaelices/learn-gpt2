"""Problem 5: Causal Masking

Learning objectives:
- Understand autoregressive language modeling
- Learn to prevent attention to future tokens
- Implement triangular causal mask
- Apply masking before softmax

TODO: Implement the CausalMultiHeadAttention class below.
"""

import torch.nn as nn
from torch import Tensor


class CausalMultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    This prevents positions from attending to subsequent positions,
    which is essential for autoregressive language modeling (like GPT-2).
    """

    def __init__(self, n_embd: int, n_head: int, n_positions: int = 1024, dropout: float = 0.1) -> None:
        """Initialize causal multi-head attention.

        Args:
            n_embd: Embedding dimension (must be divisible by n_head).
            n_head: Number of attention heads.
            n_positions: Maximum sequence length for causal mask.
            dropout: Dropout probability.

        Hints:
            - Start by copying your MultiHeadAttention implementation from Problem 4
            - Create a causal mask: torch.tril(torch.ones(n_positions, n_positions))
            - Shape the mask for broadcasting: (1, 1, n_positions, n_positions)
            - Register it as a buffer: self.register_buffer("bias", mask)
            - The buffer moves with the model to GPU/CPU automatically
        """
        super().__init__()
        # TODO: Validate n_embd is divisible by n_head
        # TODO: Store n_embd, n_head, head_dim, and n_positions

        # TODO: Create combined Q,K,V projection
        # TODO: Create output projection
        # TODO: Create dropout layers (attention and residual)

        # TODO: Create causal mask
        # 1. Create lower-triangular matrix: torch.tril(torch.ones(n_positions, n_positions))
        # 2. Reshape for broadcasting: .view(1, 1, n_positions, n_positions)
        # 3. Register as buffer: self.register_buffer("bias", mask)

        raise NotImplementedError("Complete the __init__ method")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        Hints:
            This is similar to Problem 4, but with one key addition:
            - After computing attention scores, apply the causal mask
            - Slice the mask to match current seq_len: self.bias[:, :, :seq_len, :seq_len]
            - Use masked_fill: attn_scores.masked_fill(mask == 0, float('-inf'))
            - Apply this BEFORE softmax
        """
        # TODO: Get batch_size and seq_len from x

        # TODO: Project to Q, K, V using combined projection
        # TODO: Split into Q, K, V

        # TODO: Reshape for multi-head attention
        # Shape each to: (batch_size, n_head, seq_len, head_dim)

        # TODO: Compute attention scores: (Q @ K^T) / sqrt(head_dim)
        # Shape: (batch_size, n_head, seq_len, seq_len)

        # TODO: Apply causal mask
        # 1. Get the mask slice for current seq_len: self.bias[:, :, :seq_len, :seq_len]
        # 2. Set masked positions to -inf: attn_scores.masked_fill(mask == 0, float('-inf'))
        # This ensures future positions get 0 attention weight after softmax

        # TODO: Apply softmax to get attention weights

        # TODO: Apply dropout to attention weights

        # TODO: Apply attention to values: attn_weights @ V

        # TODO: Concatenate heads back together
        # Transpose to (batch, seq_len, n_head, head_dim)
        # Reshape to (batch, seq_len, n_embd)

        # TODO: Apply output projection and dropout

        # TODO: Return output
        raise NotImplementedError("Complete the forward method")
