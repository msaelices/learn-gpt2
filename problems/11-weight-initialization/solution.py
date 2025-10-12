"""Problem 11: Weight Initialization - Solution

This solution adds proper weight initialization to the GPT-2 model.
"""

import sys
sys.path.append("../09-gpt2-config")
sys.path.append("../10-full-model")

import torch
import torch.nn as nn
from torch import Tensor

# Import from previous problems
import importlib.util
spec = importlib.util.spec_from_file_location("gpt2_config", "../09-gpt2-config/solution.py")
gpt2_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt2_config)
GPT2Config = gpt2_config.GPT2Config

spec = importlib.util.spec_from_file_location("transformer_block", "../08-transformer-block/solution.py")
transformer_block = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_block)
TransformerBlock = transformer_block.TransformerBlock


class GPT2Model(nn.Module):
    """Complete GPT-2 language model with proper weight initialization.

    This model includes proper weight initialization following GPT-2's approach:
    - Linear layers: normal(0, 0.02) for weights, zeros for biases
    - Embedding layers: normal(0, 0.02) for weights
    - LayerNorm: ones for weights, zeros for biases
    """

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 model with proper weight initialization.

        Args:
            config: GPT2Config object containing all hyperparameters.
        """
        super().__init__()
        self.config = config

        # Token embeddings: maps vocabulary indices to embedding vectors
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings: maps position indices to embedding vectors
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        # Embedding dropout for regularization
        self.drop = nn.Dropout(config.embd_pdrop)

        # Stack of transformer blocks
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
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Language model head: projects embeddings back to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output projection
        self.lm_head.weight = self.wte.weight

        # Apply weight initialization to all modules
        # This must be called AFTER all modules are created
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different layer types.

        This method is recursively applied to all modules in the model.
        It implements GPT-2's initialization strategy:
        - Small random weights (std=0.02) for linear and embedding layers
        - Zero biases for linear layers
        - Ones for LayerNorm weights, zeros for LayerNorm biases

        Args:
            module: The module to initialize. This is called for every
                   module in the model via self.apply().

        Note:
            The std=0.02 is specifically chosen for GPT-2 and has been
            empirically validated for transformer models. Other architectures
            might use different values (e.g., Xavier/Glorot or He initialization).
        """
        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with small random values
            # Normal distribution: mean=0.0, std=0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # Initialize biases to zero (if they exist)
            # Note: Some layers (like lm_head) don't have biases
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights with small random values
            # Same distribution as linear layers for consistency
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm starts with no scaling (weight=1) and no shift (bias=0)
            # This means initially: output = (input - mean) / std
            # As training progresses, the model learns optimal scale and shift
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through the GPT-2 model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        # Get input dimensions
        batch_size, seq_len = input_ids.size()

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)  # Shape: (1, seq_len)

        # Get embeddings
        token_embeddings = self.wte(input_ids)  # (B, T, d)
        position_embeddings = self.wpe(position_ids)  # (1, T, d) broadcasts to (B, T, d)

        # Combine embeddings and apply dropout
        x = token_embeddings + position_embeddings
        x = self.drop(x)

        # Pass through transformer blocks sequentially
        for block in self.h:
            x = block(x)

        # Apply final layer normalization
        x = self.ln_f(x)

        # Project to vocabulary to get logits
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
