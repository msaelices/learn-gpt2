"""Solution for Problem 3: Scaled Dot-Product Attention

This solution implements scaled dot-product attention, which is the standard
attention mechanism used in transformers. The scaling factor (1/√d_k) prevents
the softmax from saturating when the embedding dimension is large.
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

    The key difference from SimpleAttention is the division by √d_k,
    which normalizes the variance of the dot products.

    Attributes:
        query: Linear layer to project input to queries
        key: Linear layer to project input to keys
        value: Linear layer to project input to values
    """

    def __init__(self, n_embd: int = 768) -> None:
        """Initialize attention projections.

        Args:
            n_embd: Embedding dimension. All projections maintain this dimension.
        """
        super().__init__()

        # Query projection: Transforms input to "what am I looking for?"
        self.query = nn.Linear(n_embd, n_embd)

        # Key projection: Transforms input to "what do I offer?"
        self.key = nn.Linear(n_embd, n_embd)

        # Value projection: Transforms input to "what information do I contain?"
        self.value = nn.Linear(n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: compute scaled self-attention.

        This implements scaled dot-product attention:
        - Each position attends to all positions (including itself)
        - Scores are scaled by 1/√d_k for numerical stability
        - Attention weights determine how much each position contributes
        - Output is a weighted combination of values

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
                Represents context-aware embeddings where each position
                has incorporated information from other positions
        """
        # Step 1: Project input to queries, keys, and values
        # All have shape: (batch_size, seq_len, n_embd)
        q = self.query(x)  # "What am I looking for?"
        k = self.key(x)  # "What do I offer?"
        v = self.value(x)  # "What information do I have?"

        # Step 2: Compute attention scores
        # Shape: (batch_size, seq_len, seq_len)
        attn_scores = q @ k.transpose(-2, -1)

        # Step 3: Scale the attention scores by √d_k
        # This is the KEY DIFFERENCE from SimpleAttention!
        # Why? When d_k is large, dot products grow large in magnitude,
        # pushing softmax into regions with extremely small gradients.
        # Dividing by √d_k normalizes the variance to ~1.
        scale = q.size(-1) ** -0.5  # 1 / √d_k
        attn_scores = attn_scores * scale

        # Step 4: Convert scores to probabilities using softmax
        # Shape: (batch_size, seq_len, seq_len)
        # Each row sums to 1, representing a probability distribution over positions
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Step 5: Apply attention weights to values
        # Shape: (batch_size, seq_len, n_embd)
        # output[b, i, :] = weighted combination of all value vectors
        output = attn_weights @ v

        return output


# Example usage
if __name__ == "__main__":
    # Create attention module
    attention = ScaledAttention(n_embd=512)

    # Create sample input (batch_size=2, seq_len=5, n_embd=512)
    x = torch.randn(2, 5, 512)

    print("Input shape:", x.shape)

    # Forward pass
    output = attention(x)

    print("Output shape:", output.shape)
    print("Expected shape: (2, 5, 512)")

    # Compare scaled vs unscaled attention scores
    with torch.no_grad():
        q = attention.query(x)
        k = attention.key(x)

        # Unscaled scores
        unscaled_scores = q @ k.transpose(-2, -1)
        print("\nUnscaled attention scores:")
        print(f"  Range: {unscaled_scores.min().item():.2f} to {unscaled_scores.max().item():.2f}")
        print(f"  Std: {unscaled_scores.std().item():.2f}")

        # Scaled scores
        scale = q.size(-1) ** -0.5
        scaled_scores = unscaled_scores * scale
        print(f"\nScaled attention scores (÷ √{q.size(-1)} = {scale:.4f}):")
        print(f"  Range: {scaled_scores.min().item():.2f} to {scaled_scores.max().item():.2f}")
        print(f"  Std: {scaled_scores.std().item():.2f}")

        # Attention weights
        attn_weights = torch.softmax(scaled_scores, dim=-1)
        print(f"\nAttention weights shape: {attn_weights.shape}")
        print(f"Sum of attention weights (should be ~1.0): {attn_weights[0, 0, :].sum().item():.6f}")

    print("\n✅ Scaled attention working correctly!")
