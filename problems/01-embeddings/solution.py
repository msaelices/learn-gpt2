"""Solution for Problem 1: Token & Position Embeddings

This solution implements the embedding layer for GPT-2, combining
token embeddings and position embeddings.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Embeddings(nn.Module):
    """Token and position embeddings for GPT-2.

    This module combines token embeddings and position embeddings to create
    the input representation for the transformer.

    In GPT-2, embeddings serve two purposes:
    1. Token embeddings: Map each vocabulary item to a learned vector
    2. Position embeddings: Add positional information to each token

    The final embedding is simply the sum of token and position embeddings.

    Attributes:
        wte: Token embedding layer (vocab_size → n_embd)
        wpe: Position embedding layer (n_positions → n_embd)
        drop: Dropout layer for regularization
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
    ) -> None:
        """Initialize token and position embeddings.

        Args:
            vocab_size: Size of the vocabulary (number of possible tokens).
                       For GPT-2, this is 50,257.
            n_positions: Maximum sequence length supported.
                        For GPT-2, this is 1024.
            n_embd: Dimension of the embedding vectors.
                   For GPT-2 small, this is 768.
            embd_pdrop: Dropout probability for embedding dropout.
                       For GPT-2, this is 0.1.
        """
        super().__init__()

        # Token embeddings: map vocabulary index to embedding vector
        # Shape: (vocab_size, n_embd)
        # Each row is the embedding for a specific token
        self.wte = nn.Embedding(vocab_size, n_embd)

        # Position embeddings: map position index to embedding vector
        # Shape: (n_positions, n_embd)
        # Each row is the embedding for a specific position
        self.wpe = nn.Embedding(n_positions, n_embd)

        # Dropout for regularization
        # Applied after combining token and position embeddings
        self.drop = nn.Dropout(embd_pdrop)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass: convert token IDs to embeddings.

        The forward pass performs these steps:
        1. Extract batch size and sequence length from input
        2. Create position indices [0, 1, 2, ..., seq_len-1]
        3. Look up token embeddings for each input token
        4. Look up position embeddings for each position
        5. Add token and position embeddings together
        6. Apply dropout

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
                      Values should be integers in range [0, vocab_size)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, n_embd)
        """
        # Get batch size and sequence length from input
        batch_size, seq_len = input_ids.size()

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # Important: Must be on the same device as input_ids
        # Shape: (seq_len,)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)

        # Expand position_ids to have a batch dimension
        # Shape: (1, seq_len) → (batch_size, seq_len)
        # This allows broadcasting when we look up embeddings
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Look up token embeddings
        # Input shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, n_embd)
        token_embeddings = self.wte(input_ids)

        # Look up position embeddings
        # Input shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, n_embd)
        position_embeddings = self.wpe(position_ids)

        # Combine token and position embeddings via element-wise addition
        # Both have shape (batch_size, seq_len, n_embd)
        # Result has shape (batch_size, seq_len, n_embd)
        embeddings = token_embeddings + position_embeddings

        # Apply dropout for regularization
        # In training mode: randomly zero some elements
        # In eval mode: pass through unchanged
        embeddings = self.drop(embeddings)

        return embeddings


# Example usage
if __name__ == "__main__":
    # Create embeddings module with GPT-2 small configuration
    embeddings = Embeddings(
        vocab_size=50257,  # GPT-2 vocabulary size
        n_positions=1024,  # GPT-2 max sequence length
        n_embd=768,  # GPT-2 small embedding dimension
        embd_pdrop=0.1,  # GPT-2 dropout rate
    )

    # Create sample input (batch_size=2, seq_len=10)
    input_ids = torch.randint(0, 50257, (2, 10))

    # Forward pass
    output = embeddings(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print("Expected output shape: (2, 10, 768)")

    # Verify shapes
    assert output.shape == (2, 10, 768), "Output shape mismatch!"
    print("\n✅ Embeddings working correctly!")

    # Count parameters
    total_params = sum(p.numel() for p in embeddings.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  - Token embeddings: {embeddings.wte.weight.numel():,}")
    print(f"  - Position embeddings: {embeddings.wpe.weight.numel():,}")
