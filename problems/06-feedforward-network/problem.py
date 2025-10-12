"""Problem 6: Feedforward Network (MLP)

Learning objectives:
- Implement GELU activation function
- Create two-layer feedforward network
- Understand 4x expansion and projection
- Apply dropout for regularization

TODO: Implement NewGELU and FeedForward classes below.
"""

import torch
import torch.nn as nn
from torch import Tensor


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function used in GPT-2.

    This is the approximate form used in the original GPT-2 paper.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Activated tensor with the same shape as input.

        Formula:
            GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

        Hints:
            - Use torch.tanh() for the hyperbolic tangent
            - Use torch.pow(x, 3) or x**3 for the cubic term
            - √(2/π) ≈ 0.7978845608 or use torch.sqrt(torch.tensor(2.0 / torch.pi))
            - Don't forget the 0.5 multiplier at the beginning
            - The formula has three main parts:
              1. 0.5 * x (base scaling)
              2. 1 + tanh(...) (shifted tanh)
              3. √(2/π) * (x + 0.044715 * x³) (argument to tanh)
        """
        # TODO: Implement GELU approximation
        # Step 1: Compute the cubic term: 0.044715 * x³
        # Step 2: Compute the argument to tanh: √(2/π) * (x + cubic_term)
        # Step 3: Apply tanh and add 1: 1 + tanh(...)
        # Step 4: Multiply everything together: 0.5 * x * result
        raise NotImplementedError("Complete the forward method")


class FeedForward(nn.Module):
    """Position-wise feedforward network.

    A two-layer MLP that is applied independently to each position.
    This provides additional non-linear transformation capacity.
    """

    def __init__(self, n_embd: int, n_inner: int, dropout: float = 0.1) -> None:
        """Initialize feedforward network.

        Args:
            n_embd: Embedding dimension (input/output size).
            n_inner: Inner/hidden dimension (typically 4 * n_embd).
            dropout: Dropout probability.

        Hints:
            - Create two linear layers: n_embd → n_inner → n_embd
            - Create a GELU activation (use NewGELU class)
            - Create a dropout layer
            - The structure is: Linear → GELU → Linear → Dropout
        """
        super().__init__()
        # TODO: Create first linear layer (expansion): n_embd → n_inner
        # TODO: Create GELU activation
        # TODO: Create second linear layer (projection): n_inner → n_embd
        # TODO: Create dropout layer
        raise NotImplementedError("Complete the __init__ method")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through feedforward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        Hints:
            - Pass through first linear layer (expansion)
            - Apply GELU activation
            - Pass through second linear layer (projection)
            - Apply dropout
            - Return result

            The flow is:
            x → Linear₁ → GELU → Linear₂ → Dropout → output
        """
        # TODO: Apply first linear transformation (expansion)
        # TODO: Apply GELU activation
        # TODO: Apply second linear transformation (projection)
        # TODO: Apply dropout
        # TODO: Return output
        raise NotImplementedError("Complete the forward method")
