"""Problem 10: Full GPT-2 Model Assembly

Learning objectives:
- Assemble all components into complete model
- Stack transformer blocks
- Implement language modeling head
- Handle weight tying

TODO: Implement the GPT2Model class below.
"""

import sys

sys.path.append("../09-gpt2-config")
sys.path.append("../08-transformer-block")

import torch.nn as nn
from solution import GPT2Config  # From Problem 9
from torch import Tensor


class GPT2Model(nn.Module):
    """Complete GPT-2 language model.

    Architecture:
        1. Token + Position Embeddings
        2. Embedding Dropout
        3. Stack of N Transformer Blocks
        4. Final Layer Normalization
        5. Language Model Head (tied with token embeddings)
    """

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config object with model hyperparameters.

        Hints:
            - Store the config as self.config
            - Create token embeddings: nn.Embedding(vocab_size, n_embd)
            - Create position embeddings: nn.Embedding(n_positions, n_embd)
            - Create embedding dropout: nn.Dropout(config.embd_pdrop)
            - Create transformer blocks: nn.ModuleList([TransformerBlock(...) for _ in range(n_layer)])
            - Create final layer norm: nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
            - Create language model head: nn.Linear(n_embd, vocab_size, bias=False)
            - Tie weights: self.lm_head.weight = self.wte.weight
        """
        super().__init__()
        # TODO: Store config
        # TODO: Create token embedding layer (wte)
        # TODO: Create position embedding layer (wpe)
        # TODO: Create embedding dropout
        # TODO: Create stack of transformer blocks (use nn.ModuleList)
        # TODO: Create final layer normalization
        # TODO: Create language model head
        # TODO: Tie lm_head weights with wte weights
        raise NotImplementedError("Complete the __init__ method")

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through GPT-2.

        Args:
            input_ids: Input token indices of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).

        Hints:
            - Get batch_size and seq_len from input_ids.shape
            - Create position IDs: torch.arange(0, seq_len, device=input_ids.device)
            - Get token embeddings: self.wte(input_ids)
            - Get position embeddings: self.wpe(position_ids)
            - Combine embeddings and apply dropout
            - Pass through each transformer block sequentially
            - Apply final layer norm
            - Project to vocabulary using lm_head
            - Return logits (don't apply softmax here)
        """
        # TODO: Get batch_size and seq_len from input shape
        # TODO: Create position indices [0, 1, 2, ..., seq_len-1]
        # TODO: Get token embeddings
        # TODO: Get position embeddings
        # TODO: Combine token + position embeddings
        # TODO: Apply embedding dropout
        # TODO: Pass through all transformer blocks sequentially
        # TODO: Apply final layer normalization
        # TODO: Project to vocabulary (get logits)
        # TODO: Return logits
        raise NotImplementedError("Complete the forward method")
