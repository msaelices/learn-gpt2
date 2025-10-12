"""Problem 7: Layer Normalization & Residuals - Solution

This solution implements the pre-norm residual connection pattern used in GPT-2.
"""

import torch.nn as nn
from torch import Tensor
from typing import Callable


class ResidualConnection(nn.Module):
    """Pre-norm residual connection.

    Applies layer normalization, then the sublayer, then adds the residual connection.
    This is the pattern used in GPT-2 (pre-norm architecture).

    Pattern:
        output = x + sublayer(layer_norm(x))

    This provides two key benefits:
    1. Layer normalization stabilizes activations
    2. Residual connection provides direct gradient path
    """

    def __init__(self, n_embd: int, dropout: float = 0.1, layer_norm_epsilon: float = 1e-5) -> None:
        """Initialize residual connection with layer normalization.

        Args:
            n_embd: Embedding dimension to normalize.
            dropout: Dropout probability (applied after sublayer).
            layer_norm_epsilon: Small constant for numerical stability in layer norm.
        """
        super().__init__()
        # Layer normalization normalizes across the embedding dimension
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        # Dropout for regularization (applied after sublayer output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """Apply pre-norm residual connection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            sublayer: A function/module that takes x and returns same shape.
                     This could be attention or feedforward network.

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        The transformation flow:
            1. Normalize input: ln(x)
            2. Apply sublayer: sublayer(ln(x))
            3. Apply dropout: dropout(sublayer(ln(x)))
            4. Add residual: x + dropout(sublayer(ln(x)))
        """
        # Pre-norm: apply layer norm before sublayer
        normalized = self.ln(x)
        # Pass through sublayer (attention or feedforward)
        sublayer_output = sublayer(normalized)
        # Apply dropout for regularization
        sublayer_output = self.dropout(sublayer_output)
        # Add residual connection (element-wise addition)
        return x + sublayer_output
