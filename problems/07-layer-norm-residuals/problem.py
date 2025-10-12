"""Problem 7: Layer Normalization & Residuals

Learning objectives:
- Understand layer normalization and why it's needed
- Learn about residual connections (skip connections)
- Implement pre-norm architecture
- Understand gradient flow through deep networks

TODO: Implement the ResidualConnection class below.
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
    """

    def __init__(self, n_embd: int, dropout: float = 0.1, layer_norm_epsilon: float = 1e-5) -> None:
        """Initialize residual connection with layer normalization.

        Args:
            n_embd: Embedding dimension to normalize.
            dropout: Dropout probability (applied after sublayer).
            layer_norm_epsilon: Small constant for numerical stability in layer norm.

        Hints:
            - Use nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
            - Create a dropout layer with nn.Dropout(dropout)
            - Layer norm normalizes across the last dimension (embedding dimension)
        """
        super().__init__()
        # TODO: Create layer normalization
        # TODO: Create dropout layer
        raise NotImplementedError("Complete the __init__ method")

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """Apply pre-norm residual connection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            sublayer: A function/module that takes x and returns same shape.
                     This could be attention or feedforward network.

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        Hints:
            - Apply layer normalization to x first (pre-norm)
            - Pass normalized x through the sublayer
            - Apply dropout to the sublayer output
            - Add the residual connection: x + dropout(sublayer(ln(x)))
            - The pattern is: x + dropout(sublayer(layer_norm(x)))
        """
        # TODO: Apply layer normalization to input
        # TODO: Pass through sublayer
        # TODO: Apply dropout
        # TODO: Add residual connection (original x + sublayer output)
        # TODO: Return result
        raise NotImplementedError("Complete the forward method")
