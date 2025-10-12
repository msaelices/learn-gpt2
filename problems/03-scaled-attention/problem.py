"""Problem 3: Scaled Dot-Product Attention

Learning objectives:
- Understand the importance of scaling in attention mechanisms
- Implement scaled dot-product attention
- Learn about numerical stability and gradient flow

TODO: Complete the ScaledAttention class by adding the scaling factor.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ScaledAttention(nn.Module):
    """Scaled dot-product attention mechanism.

    This is the standard attention mechanism used in transformers,
    with scaling to prevent vanishing gradients when d_k is large.

    The attention mechanism computes:
    1. Q, K, V projections from input
    2. Attention scores = Q @ K^T / √d_k  (scaled!)
    3. Attention weights = softmax(scores)
    4. Output = weights @ V

    Attributes:
        query: Linear layer to project input to queries
        key: Linear layer to project input to keys
        value: Linear layer to project input to values
    """

    def __init__(self, n_embd: int = 768) -> None:
        """Initialize attention projections.

        Args:
            n_embd: Embedding dimension. All projections maintain this dimension.

        Hints:
            - This is identical to SimpleAttention from Problem 2
            - Create query, key, and value linear layers
        """
        super().__init__()

        # TODO: Create query projection layer
        # Hint: self.query = nn.Linear(n_embd, n_embd)
        raise NotImplementedError("Create query projection layer")

        # TODO: Create key projection layer
        # Hint: self.key = nn.Linear(n_embd, n_embd)
        raise NotImplementedError("Create key projection layer")

        # TODO: Create value projection layer
        # Hint: self.value = nn.Linear(n_embd, n_embd)
        raise NotImplementedError("Create value projection layer")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: compute scaled self-attention.

        This is nearly identical to SimpleAttention, with one key difference:
        we divide attention scores by √d_k before applying softmax.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)

        Hints:
            - Get Q, K, V projections (same as Problem 2)
            - Compute attention scores: Q @ K^T
            - NEW: Scale the scores by dividing by √d_k
            - Apply softmax to get weights
            - Multiply weights by V
        """
        # TODO: Project input to queries, keys, and values
        # Hint: Same as Problem 2
        # q = self.query(x)
        # k = self.key(x)
        # v = self.value(x)
        raise NotImplementedError("Project input to Q, K, V")

        # TODO: Compute attention scores (Q @ K^T)
        # Hint: attn_scores = q @ k.transpose(-2, -1)
        raise NotImplementedError("Compute attention scores")

        # TODO: Scale the attention scores by √d_k
        # Hint 1: Get d_k from the last dimension of q: d_k = q.size(-1)
        # Hint 2: Compute scale factor: scale = d_k ** -0.5 (same as 1/√d_k)
        # Hint 3: Multiply scores by scale: attn_scores = attn_scores * scale
        # Alternative: attn_scores = attn_scores / math.sqrt(d_k)
        raise NotImplementedError("Scale attention scores by √d_k")

        # TODO: Apply softmax to get attention weights
        # Hint: attn_weights = torch.softmax(attn_scores, dim=-1)
        raise NotImplementedError("Apply softmax to get attention weights")

        # TODO: Apply attention weights to values
        # Hint: output = attn_weights @ v
        raise NotImplementedError("Apply attention weights to values")

        # TODO: Return the output
        raise NotImplementedError("Return output")


# Example usage (uncomment to test):
# if __name__ == "__main__":
#     # Create attention module
#     attention = ScaledAttention(n_embd=64)
#
#     # Create sample input
#     x = torch.randn(2, 5, 64)
#
#     print("Input shape:", x.shape)
#
#     # Forward pass
#     output = attention(x)
#
#     print("Output shape:", output.shape)
#     print("Expected shape: (2, 5, 64)")
#
#     # Compare with unscaled attention
#     with torch.no_grad():
#         q = attention.query(x)
#         k = attention.key(x)
#
#         # Unscaled scores
#         unscaled_scores = q @ k.transpose(-2, -1)
#         print(f"\nUnscaled score range: {unscaled_scores.min().item():.2f} to {unscaled_scores.max().item():.2f}")
#
#         # Scaled scores
#         scale = q.size(-1) ** -0.5
#         scaled_scores = unscaled_scores * scale
#         print(f"Scaled score range: {scaled_scores.min().item():.2f} to {scaled_scores.max().item():.2f}")
#         print(f"Scaling factor: {scale:.4f} (1/√{q.size(-1)})")
