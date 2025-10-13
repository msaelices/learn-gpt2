"""Problem 1: Token & Position Embeddings

Learning objectives:
- Understand embeddings and their role in language models
- Implement token embeddings (vocabulary → vectors)
- Implement position embeddings (position → vectors)
- Combine embeddings and apply dropout

TODO: Complete the Embeddings class below by implementing __init__ and forward methods.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Embeddings(nn.Module):
    """Token and position embeddings for GPT-2.

    This module combines token embeddings and position embeddings to create
    the input representation for the transformer.

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
            n_positions: Maximum sequence length supported.
            n_embd: Dimension of the embedding vectors.
            embd_pdrop: Dropout probability for embedding dropout.

        Hints:
            - Use nn.Embedding(num_embeddings, embedding_dim) for both token and position embeddings
            - Token embeddings: vocab_size → n_embd
            - Position embeddings: n_positions → n_embd
            - Use nn.Dropout for regularization
        """
        super().__init__()

        self.wte = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.wpe = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)

        self.dropout = nn.Dropout(p=embd_pdrop)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass: convert token IDs to embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Combined embeddings of shape (batch_size, seq_len, n_embd)

        Hints:
            - Get batch_size and seq_len from input_ids.size()
            - Create position IDs: torch.arange(0, seq_len)
            - Position IDs must be on the same device as input_ids
            - Expand position IDs to match batch size
            - Lookup token embeddings: self.wte(input_ids)
            - Lookup position embeddings: self.wpe(position_ids)
            - Add them together element-wise
            - Apply dropout to the result
        """
        batch_size, seq_len = input_ids.size()

        # TODO: Create position indices [0, 1, 2, ..., seq_len-1]
        # Important: Make sure position_ids is on the same device as input_ids!
        position_ids = torch.arange(
            0,
            seq_len,
            device=input_ids.device,
        )

        # TODO: Expand position_ids to have a batch dimension
        # Shape should go from (seq_len,) to (batch_size, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # TODO: Get token embeddings
        # Expected shape: (batch_size, seq_len, n_embd)
        token_embeddings = self.wte(input_ids)

        # TODO: Get position embeddings
        # Expected shape: (batch_size, seq_len, n_embd)
        position_embeddings = self.wpe(position_ids)

        # TODO: Combine token and position embeddings by adding them
        embeddings = token_embeddings + position_embeddings

        # TODO: Apply dropout
        return self.dropout(embeddings)


# Example usage (uncomment to test):
# if __name__ == "__main__":
#     # Create embeddings module
#     embeddings = Embeddings(vocab_size=50257, n_positions=1024, n_embd=768)
#
#     # Create sample input (batch_size=2, seq_len=10)
#     input_ids = torch.randint(0, 50257, (2, 10))
#
#     # Forward pass
#     output = embeddings(input_ids)
#
#     print(f"Input shape: {input_ids.shape}")
#     print(f"Output shape: {output.shape}")
#     print("Expected output shape: (2, 10, 768)")
