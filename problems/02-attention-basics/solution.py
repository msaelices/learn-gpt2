"""Solution for Problem 2: Attention Basics

This solution implements a basic single-head self-attention mechanism
without scaling. This helps understand the core attention concept before
adding complexity.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SimpleAttention(nn.Module):
    """Basic single-head self-attention without scaling.

    This is a simplified attention mechanism that demonstrates the core
    Query-Key-Value concept without the scaling factor. The scaling
    will be added in the next problem for numerical stability.

    The attention mechanism computes:
    1. Q, K, V projections from input
    2. Attention scores = Q @ K^T
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
        """
        super().__init__()

        # Query projection: Transforms input to "what am I looking for?"
        self.query = nn.Linear(n_embd, n_embd)

        # Key projection: Transforms input to "what do I offer?"
        self.key = nn.Linear(n_embd, n_embd)

        # Value projection: Transforms input to "what information do I contain?"
        self.value = nn.Linear(n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: compute self-attention.

        This implements the core attention mechanism:
        - Each position attends to all positions (including itself)
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
        # attn_scores[b, i, j] = similarity between query at position i and key at position j
        # The transpose swaps seq_len and n_embd dimensions: (batch, seq_len, n_embd) -> (batch, n_embd, seq_len)
        # Then @ performs batch matrix multiply: (batch, seq_len, n_embd) @ (batch, n_embd, seq_len)
        # Result: (batch, seq_len, seq_len)
        attn_scores = q @ k.transpose(-2, -1)

        # Step 3: Convert scores to probabilities using softmax
        # Shape: (batch_size, seq_len, seq_len)
        # Each row sums to 1, representing a probability distribution over positions
        # attn_weights[b, i, :] = probability distribution of where position i attends
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Step 4: Apply attention weights to values
        # Shape: (batch_size, seq_len, n_embd)
        # output[b, i, :] = weighted combination of all value vectors
        # where weights come from attn_weights[b, i, :]
        output = attn_weights @ v

        return output


# Example usage
if __name__ == "__main__":
    # Create attention module
    attention = SimpleAttention(n_embd=64)

    # Create sample input (batch_size=2, seq_len=5, n_embd=64)
    x = torch.randn(2, 5, 64)

    print("Input shape:", x.shape)

    # Forward pass
    output = attention(x)

    print("Output shape:", output.shape)
    print("Expected shape: (2, 5, 64)")

    # Verify attention weights sum to 1
    with torch.no_grad():
        q = attention.query(x)
        k = attention.key(x)
        attn_scores = q @ k.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        print(f"\nAttention weights shape: {attn_weights.shape}")
        print(f"Sum of attention weights (should be ~1.0):")
        print(attn_weights[0, 0, :].sum().item())

    print("\n✅ Simple attention working correctly!")
