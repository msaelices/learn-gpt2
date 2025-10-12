"""Problem 6: Feedforward Network (MLP) - Solution

This solution implements the position-wise feedforward network,
extracted from the reference implementation in src/gpt2/model.py.
"""

import torch
import torch.nn as nn
from torch import Tensor


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function used in GPT-2.

    This is the approximate form used in the original GPT-2 paper and HuggingFace.
    GELU (Gaussian Error Linear Unit) is a smooth, non-monotonic activation function
    that empirically performs better than ReLU for language models.

    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply NewGELU activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Activated tensor with the same shape as input.

        Formula:
            GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        """
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
    """Position-wise feedforward network.

    A two-layer MLP that is applied independently to each position in the sequence.
    The first layer expands the dimension by 4x (from n_embd to n_inner),
    applies GELU activation, then the second layer projects back to n_embd.

    This provides additional non-linear transformation capacity to the model.
    """

    def __init__(self, n_embd: int, n_inner: int, dropout: float = 0.1) -> None:
        """Initialize feedforward network.

        Args:
            n_embd: Embedding dimension (input/output size).
            n_inner: Inner/hidden dimension (typically 4 * n_embd for GPT-2).
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        # Two-layer MLP with NewGELU activation
        # First layer: expand from n_embd to n_inner (typically 4x)
        self.c_fc = nn.Linear(n_embd, n_inner)
        # Second layer: project back from n_inner to n_embd
        self.c_proj = nn.Linear(n_inner, n_embd)
        # Dropout for regularization (applied after the second layer)
        self.dropout = nn.Dropout(dropout)
        # GELU activation (applied after first layer)
        self.act = NewGELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through feedforward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        The transformation flow:
            x → Linear(n_embd → n_inner) → GELU → Linear(n_inner → n_embd) → Dropout
        """
        # Expand dimension: (batch, seq_len, n_embd) → (batch, seq_len, n_inner)
        x = self.c_fc(x)
        # Apply non-linear activation
        x = self.act(x)
        # Project back: (batch, seq_len, n_inner) → (batch, seq_len, n_embd)
        x = self.c_proj(x)
        # Apply dropout for regularization
        x = self.dropout(x)
        return x
