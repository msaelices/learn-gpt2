"""Problem 4: Multi-Head Attention

Learning objectives:
- Understand why multiple attention heads improve model capacity
- Learn to split embeddings across heads for parallel attention
- Implement efficient combined Q,K,V projection
- Master head concatenation and output projection

TODO: Implement the MultiHeadAttention class below.
"""

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1) -> None:
        """Initialize multi-head attention.

        Args:
            n_embd: Embedding dimension (must be divisible by n_head).
            n_head: Number of attention heads.
            dropout: Dropout probability for attention weights and output.

        Hints:
            - Validate that n_embd is divisible by n_head
            - Calculate head_dim = n_embd // n_head
            - Use nn.Linear(n_embd, 3 * n_embd) for combined Q,K,V projection
            - Use nn.Linear(n_embd, n_embd) for output projection
            - Create two dropout layers: one for attention, one for output
        """
        super().__init__()
        # TODO: Validate n_embd is divisible by n_head
        # TODO: Store n_embd, n_head, and head_dim as instance variables
        # TODO: Create combined Q,K,V projection (more efficient than 3 separate ones)
        # TODO: Create output projection
        # TODO: Create dropout layers
        raise NotImplementedError("Complete the __init__ method")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).

        Hints:
            Step-by-step process:
            1. Get batch_size, seq_len from x.size()
            2. Project x through combined Q,K,V linear layer
            3. Split the result into Q, K, V using tensor.split(n_embd, dim=2)
            4. Reshape each to (batch_size, seq_len, n_head, head_dim)
            5. Transpose to (batch_size, n_head, seq_len, head_dim) for efficient computation
            6. Compute attention scores: (Q @ K^T) / sqrt(head_dim)
            7. Apply softmax to get attention weights
            8. Apply dropout to attention weights
            9. Compute output: attention_weights @ V
            10. Transpose back to (batch_size, seq_len, n_head, head_dim)
            11. Reshape/concatenate heads to (batch_size, seq_len, n_embd)
            12. Apply output projection
            13. Apply output dropout
            14. Return result
        """
        # TODO: Extract batch_size and seq_len from x

        # TODO: Step 1: Project to Q, K, V using combined projection
        # Shape after projection: (batch_size, seq_len, 3 * n_embd)

        # TODO: Step 2: Split into Q, K, V
        # Each should have shape: (batch_size, seq_len, n_embd)

        # TODO: Step 3: Reshape for multi-head attention
        # Reshape Q, K, V to: (batch_size, seq_len, n_head, head_dim)
        # Then transpose to: (batch_size, n_head, seq_len, head_dim)

        # TODO: Step 4: Compute scaled dot-product attention scores
        # Shape: (batch_size, n_head, seq_len, seq_len)
        # Remember to scale by sqrt(head_dim)!

        # TODO: Step 5: Apply softmax to get attention weights

        # TODO: Step 6: Apply dropout to attention weights

        # TODO: Step 7: Apply attention to values
        # Shape: (batch_size, n_head, seq_len, head_dim)

        # TODO: Step 8: Concatenate heads back together
        # Transpose to (batch_size, seq_len, n_head, head_dim)
        # Then reshape to (batch_size, seq_len, n_embd)
        # Don't forget .contiguous() before .view()!

        # TODO: Step 9: Apply output projection

        # TODO: Step 10: Apply output dropout

        # TODO: Return the result
        raise NotImplementedError("Complete the forward method")
