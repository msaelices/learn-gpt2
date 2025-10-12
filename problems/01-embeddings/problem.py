"""Problem 1: Token & Position Embeddings

Learning objectives:
- Understand embeddings and their role in language models
- Implement token embeddings (vocabulary → vectors)
- Implement position embeddings (position → vectors)
- Combine embeddings and apply dropout

TODO: Complete the Embeddings class below by implementing __init__ and forward methods.
"""

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

        # TODO: Create token embedding layer (wte)
        # This should map vocab_size token IDs to n_embd dimensional vectors
        # Hint: self.wte = nn.Embedding(...)
        raise NotImplementedError("Create token embedding layer (wte)")

        # TODO: Create position embedding layer (wpe)
        # This should map n_positions position indices to n_embd dimensional vectors
        # Hint: self.wpe = nn.Embedding(...)
        raise NotImplementedError("Create position embedding layer (wpe)")

        # TODO: Create dropout layer
        # Hint: self.drop = nn.Dropout(...)
        raise NotImplementedError("Create dropout layer")

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
        # TODO: Get the batch size and sequence length from input_ids
        # Hint: batch_size, seq_len = input_ids.size()
        raise NotImplementedError("Get batch_size and seq_len from input_ids")

        # TODO: Create position indices [0, 1, 2, ..., seq_len-1]
        # Hint: Use torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        # Important: Make sure position_ids is on the same device as input_ids!
        raise NotImplementedError("Create position indices")

        # TODO: Expand position_ids to have a batch dimension
        # Shape should go from (seq_len,) to (batch_size, seq_len)
        # Hint: position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        raise NotImplementedError("Expand position_ids to match batch size")

        # TODO: Get token embeddings
        # Hint: token_embeddings = self.wte(input_ids)
        # Expected shape: (batch_size, seq_len, n_embd)
        raise NotImplementedError("Get token embeddings")

        # TODO: Get position embeddings
        # Hint: position_embeddings = self.wpe(position_ids)
        # Expected shape: (batch_size, seq_len, n_embd)
        raise NotImplementedError("Get position embeddings")

        # TODO: Combine token and position embeddings by adding them
        # Hint: embeddings = token_embeddings + position_embeddings
        raise NotImplementedError("Combine embeddings")

        # TODO: Apply dropout
        # Hint: embeddings = self.drop(embeddings)
        raise NotImplementedError("Apply dropout")

        # TODO: Return the final embeddings
        # Expected shape: (batch_size, seq_len, n_embd)
        raise NotImplementedError("Return embeddings")


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
#     print(f"Expected output shape: (2, 10, 768)")
