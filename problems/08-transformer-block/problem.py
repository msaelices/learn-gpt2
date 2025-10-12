"""Problem 8: Complete Transformer Block

Learning objectives:
- Combine all previous components
- Implement full transformer block
- Understand information flow
- Connect attention and feedforward paths

TODO: Implement the TransformerBlock class below.
"""

import torch.nn as nn
from torch import Tensor

# Import from previous problems (you'll need to copy these or import from src/)
# For this problem, assume these classes are available:
# - MultiHeadAttention (from Problem 4/5)
# - FeedForward (from Problem 6)


class TransformerBlock(nn.Module):
    """Complete transformer block: Attention + FFN with residuals.

    This block implements the pre-norm architecture used in GPT-2:
        1. x = x + attention(layer_norm_1(x))
        2. x = x + feedforward(layer_norm_2(x))
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
            n_positions: Maximum sequence length (for causal mask).
            n_inner: Inner dimension of feedforward network. If None, defaults to 4 * n_embd.
            attn_pdrop: Attention dropout probability.
            resid_pdrop: Residual dropout probability.
            layer_norm_epsilon: Epsilon for layer normalization.

        Hints:
            - Create two separate layer norms: ln_1 and ln_2
            - Create causal multi-head attention with the given parameters
            - Create feedforward network with n_inner (default 4 * n_embd)
            - Both attention and feedforward should apply dropout
        """
        super().__init__()
        if n_inner is None:
            n_inner = 4 * n_embd

        # TODO: Create layer normalization before attention (ln_1)
        # TODO: Create causal multi-head attention
        # TODO: Create layer normalization before feedforward (ln_2)
        # TODO: Create feedforward network (MLP)
        raise NotImplementedError("Complete the __init__ method")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        Hints:
            - Apply pre-norm pattern for attention: x = x + attention(layer_norm(x))
            - Apply pre-norm pattern for feedforward: x = x + feedforward(layer_norm(x))
            - Make sure to include the residual connections (the + operation)
            - The attention layer handles its own causal masking
        """
        # TODO: Apply layer norm 1
        # TODO: Apply attention
        # TODO: Add residual connection (x = x + attention_output)
        # TODO: Apply layer norm 2
        # TODO: Apply feedforward
        # TODO: Add residual connection (x = x + feedforward_output)
        # TODO: Return output
        raise NotImplementedError("Complete the forward method")
