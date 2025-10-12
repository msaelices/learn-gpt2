"""Problem 12: Loading Pretrained Weights

Learning objectives:
- Load pretrained GPT-2 weights from HuggingFace
- Handle architecture differences
- Map state dict keys correctly
- Transpose Conv1D weights to Linear format

TODO: Implement the from_pretrained class method.
"""

import sys

sys.path.append("../09-gpt2-config")
sys.path.append("../11-weight-initialization")

# Import from previous problems
import importlib.util

import torch
import torch.nn as nn
from torch import Tensor

spec = importlib.util.spec_from_file_location("gpt2_config", "../09-gpt2-config/solution.py")
gpt2_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt2_config)
GPT2Config = gpt2_config.GPT2Config

spec = importlib.util.spec_from_file_location("gpt2_model", "../11-weight-initialization/solution.py")
gpt2_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt2_model)


class GPT2Model(nn.Module):
    """Complete GPT-2 language model with pretrained weight loading capability."""

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config object with model hyperparameters.
        """
        super().__init__()
        self.config = config

        # Import TransformerBlock
        spec = importlib.util.spec_from_file_location("transformer_block", "../08-transformer-block/solution.py")
        transformer_block = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformer_block)
        TransformerBlock = transformer_block.TransformerBlock

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # Embedding dropout
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

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different layer types."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

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

    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2") -> "GPT2Model":
        """Load pretrained GPT-2 from HuggingFace.

        Args:
            model_name: Name of the model to load. Options:
                - "gpt2" (124M parameters)
                - "gpt2-medium" (355M parameters)
                - "gpt2-large" (774M parameters)
                - "gpt2-xl" (1.5B parameters)

        Returns:
            GPT2Model with pretrained weights loaded.

        Hints:
            - Import transformers: from transformers import GPT2LMHeadModel
            - Load HF model: hf_model = GPT2LMHeadModel.from_pretrained(model_name)
            - Get HF config: hf_config = hf_model.config
            - Create our config from HF config
            - Get HF state dict: hf_state = hf_model.state_dict()
            - Map keys: remove 'transformer.' prefix
            - Transpose Conv1D weights (c_attn, c_proj, c_fc)
            - Load into our model with strict=False (lm_head is tied)
        """
        # TODO: Import transformers
        # TODO: Load HuggingFace model and config
        # TODO: Create our GPT2Config from HuggingFace config
        # TODO: Initialize our model with the config
        # TODO: Get HuggingFace state dict
        # TODO: Create our state dict with mapped keys
        # TODO: For each HF parameter:
        #   - Remove 'transformer.' prefix
        #   - Transpose if Conv1D (c_attn, c_proj, c_fc weights)
        #   - Add to our state dict
        # TODO: Load state dict into our model (strict=False)
        # TODO: Set model to eval mode
        # TODO: Return model
        raise NotImplementedError("Complete the from_pretrained method")

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
