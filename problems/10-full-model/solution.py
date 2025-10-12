"""Problem 10: Full GPT-2 Model Assembly - Solution

This solution assembles all previous components into a complete GPT-2 language model.
"""

import sys

sys.path.append("../09-gpt2-config")
sys.path.append("../08-transformer-block")

# Import GPT2Config from Problem 9
import importlib.util

import torch
import torch.nn as nn
from torch import Tensor

spec = importlib.util.spec_from_file_location("gpt2_config", "../09-gpt2-config/solution.py")
gpt2_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt2_config)
GPT2Config = gpt2_config.GPT2Config

# Import TransformerBlock from Problem 8
spec = importlib.util.spec_from_file_location(
    "transformer_block", "../08-transformer-block/solution.py"
)
transformer_block = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_block)
TransformerBlock = transformer_block.TransformerBlock


class GPT2Model(nn.Module):
    """Complete GPT-2 language model.

    This class assembles all components into a full language model:
    1. Token and position embeddings
    2. Embedding dropout
    3. Stack of N transformer blocks
    4. Final layer normalization
    5. Language modeling head (tied with token embeddings)

    The model takes token IDs as input and produces logits over the vocabulary.
    """

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config object containing all hyperparameters.
        """
        super().__init__()
        self.config = config

        # Token embeddings: maps vocabulary indices to embedding vectors
        # Shape: (vocab_size, n_embd)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings: maps position indices to embedding vectors
        # Shape: (n_positions, n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # Embedding dropout for regularization
        self.drop = nn.Dropout(config.embd_pdrop)

        # Stack of transformer blocks
        # Each block transforms (B, T, d) → (B, T, d)
        self.h = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    n_positions=config.n_positions,
                    n_inner=config.n_inner,
                    attn_pdrop=config.attn_pdrop,
                    resid_pdrop=config.resid_pdrop,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                )
                for _ in range(config.n_layer)
            ]
        )

        # Final layer normalization
        # Applied after all transformer blocks
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Language model head: projects embeddings back to vocabulary
        # Shape: (n_embd, vocab_size)
        # Note: bias=False is standard for language model heads
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output projection
        # This reduces parameters and helps the model learn better representations
        # The embedding weights are used both for:
        # 1. Converting token IDs to embeddings (input)
        # 2. Converting embeddings to logits (output)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through the GPT-2 model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
            These are unnormalized scores for each token in the vocabulary
            at each position.

        The forward pass:
            1. input_ids (B, T) → embeddings (B, T, d)
            2. Pass through N transformer blocks
            3. Apply final layer norm
            4. Project to vocabulary: (B, T, d) → (B, T, V)
        """
        # Get input dimensions
        batch_size, seq_len = input_ids.size()

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # Must be on same device as input_ids
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)  # Shape: (1, seq_len)

        # Get embeddings
        # Token embeddings: (B, T) → (B, T, d)
        token_embeddings = self.wte(input_ids)
        # Position embeddings: (1, T) → (1, T, d), broadcasts to (B, T, d)
        position_embeddings = self.wpe(position_ids)

        # Combine token and position embeddings
        # Both have shape (B, T, d), so we can add them directly
        x = token_embeddings + position_embeddings

        # Apply dropout to combined embeddings
        x = self.drop(x)

        # Pass through transformer blocks sequentially
        # Each block: (B, T, d) → (B, T, d)
        for block in self.h:
            x = block(x)

        # Apply final layer normalization
        # (B, T, d) → (B, T, d)
        x = self.ln_f(x)

        # Project to vocabulary to get logits
        # (B, T, d) → (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits

    def get_num_params(self) -> int:
        """Calculate total number of parameters.

        Returns:
            Total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def get_num_params_by_component(self) -> dict:
        """Get parameter count breakdown by component.

        Returns:
            Dictionary with parameter counts for each component.
        """
        return {
            "embeddings": sum(p.numel() for p in self.wte.parameters())
            + sum(p.numel() for p in self.wpe.parameters()),
            "transformer_blocks": sum(p.numel() for block in self.h for p in block.parameters()),
            "final_ln": sum(p.numel() for p in self.ln_f.parameters()),
            "lm_head": 0,  # Weights are tied with embeddings
            "total": self.get_num_params(),
        }
