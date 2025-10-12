"""Problem 11: Weight Initialization

Learning objectives:
- Implement proper weight initialization
- Use layer-specific initialization strategies
- Apply initialization to all modules
- Ensure training stability

TODO: Add weight initialization to the GPT2Model class.
"""

import sys

sys.path.append("../09-gpt2-config")
sys.path.append("../10-full-model")

# Import from previous problems
import importlib.util

import torch
import torch.nn as nn
from torch import Tensor

spec = importlib.util.spec_from_file_location("gpt2_config", "../09-gpt2-config/solution.py")
gpt2_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt2_config)
GPT2Config = gpt2_config.GPT2Config

spec = importlib.util.spec_from_file_location("gpt2_model", "../10-full-model/solution.py")
gpt2_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt2_model)


class GPT2Model(nn.Module):
    """Complete GPT-2 language model with proper weight initialization.

    This extends the model from Problem 10 with proper weight initialization.
    """

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 model with proper weight initialization.

        Args:
            config: GPT2Config object with model hyperparameters.

        Hints:
            - Copy the initialization from Problem 10
            - Add _init_weights method
            - Call self.apply(self._init_weights) at the END of __init__
        """
        super().__init__()
        self.config = config

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # Embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Stack of transformer blocks
        # Import TransformerBlock from Problem 8
        spec = importlib.util.spec_from_file_location("transformer_block", "../08-transformer-block/solution.py")
        transformer_block = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_block)
        TransformerBlock = transformer_block.TransformerBlock

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

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        # TODO: Apply weight initialization
        # Hint: Call self.apply(self._init_weights) here
        raise NotImplementedError("Apply weight initialization")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different layer types.

        This method is called for every module in the model via self.apply().

        Args:
            module: The module to initialize.

        Hints:
            - Use isinstance(module, nn.Linear) to check module type
            - Use nn.init.normal_(tensor, mean, std) for normal initialization
            - Use nn.init.zeros_(tensor) for zero initialization
            - Use nn.init.ones_(tensor) for ones initialization
            - Check if bias exists: if module.bias is not None
            - GPT-2 uses mean=0.0, std=0.02 for weights
        """
        # TODO: Initialize Linear layers
        #   - Weight: normal(mean=0.0, std=0.02)
        #   - Bias (if exists): zeros

        # TODO: Initialize Embedding layers
        #   - Weight: normal(mean=0.0, std=0.02)

        # TODO: Initialize LayerNorm layers
        #   - Weight: ones
        #   - Bias: zeros

        raise NotImplementedError("Complete the _init_weights method")

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through GPT-2.

        Args:
            input_ids: Input token indices of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.size()

        # Create position indices
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        # Get embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)

        # Combine and apply dropout
        x = token_embeddings + position_embeddings
        x = self.drop(x)

        # Pass through transformer blocks
        for block in self.h:
            x = block(x)

        # Apply final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    def get_num_params(self) -> int:
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_params_by_component(self) -> dict:
        """Get parameter count breakdown by component."""
        return {
            "embeddings": sum(p.numel() for p in self.wte.parameters())
            + sum(p.numel() for p in self.wpe.parameters()),
            "transformer_blocks": sum(p.numel() for block in self.h for p in block.parameters()),
            "final_ln": sum(p.numel() for p in self.ln_f.parameters()),
            "lm_head": 0,  # Weights are tied
            "total": self.get_num_params(),
        }
