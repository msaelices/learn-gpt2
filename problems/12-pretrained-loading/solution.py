"""Problem 12: Loading Pretrained Weights - Solution

This solution implements loading pretrained GPT-2 weights from HuggingFace.
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

spec = importlib.util.spec_from_file_location("transformer_block", "../08-transformer-block/solution.py")
transformer_block = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_block)
TransformerBlock = transformer_block.TransformerBlock


class GPT2Model(nn.Module):
    """Complete GPT-2 language model with pretrained weight loading capability.

    This model can be initialized from scratch or loaded from HuggingFace's
    pretrained GPT-2 checkpoints.
    """

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 model.

        Args:
            config: GPT2Config object containing all hyperparameters.
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

        This method handles the key challenges of loading weights:
        1. Mapping parameter names from HuggingFace to our implementation
        2. Handling Conv1D vs Linear layer differences (transposing weights)
        3. Preserving weight tying between embeddings and lm_head

        Args:
            model_name: Name of the model to load. Options:
                - "gpt2" (124M parameters)
                - "gpt2-medium" (355M parameters)
                - "gpt2-large" (774M parameters)
                - "gpt2-xl" (1.5B parameters)

        Returns:
            GPT2Model with pretrained weights loaded from HuggingFace.

        Example:
            >>> model = GPT2Model.from_pretrained("gpt2")
            >>> input_ids = torch.tensor([[15496, 995]])  # "Hello World"
            >>> logits = model(input_ids)
        """
        # Import transformers here so it's only required for this method
        from transformers import GPT2LMHeadModel

        print(f"Loading {model_name} from HuggingFace...")

        # Load HuggingFace model and config
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_config = hf_model.config

        # Create our config from HuggingFace config
        config = GPT2Config(
            vocab_size=hf_config.vocab_size,
            n_positions=hf_config.n_positions,
            n_embd=hf_config.n_embd,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            resid_pdrop=hf_config.resid_pdrop,
            embd_pdrop=hf_config.embd_pdrop,
            attn_pdrop=hf_config.attn_pdrop,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
        )

        print(f"Creating model with config: {config.n_layer} layers, {config.n_embd} dim")

        # Initialize our model
        model = cls(config)

        # Get HuggingFace state dict
        hf_state_dict = hf_model.state_dict()

        # Create our state dict with properly mapped and transposed weights
        our_state_dict = {}

        for hf_key, hf_param in hf_state_dict.items():
            # HuggingFace keys have 'transformer.' prefix, we don't
            if hf_key.startswith('transformer.'):
                our_key = hf_key[len('transformer.'):]
            else:
                # Skip lm_head, it's tied with wte
                continue

            # HuggingFace uses Conv1D for linear layers
            # Conv1D weights have shape (in_features, out_features)
            # PyTorch Linear weights have shape (out_features, in_features)
            # We need to transpose!

            if 'weight' in our_key and any(name in our_key for name in ['c_attn', 'c_proj', 'c_fc']):
                # These are Conv1D layers in HuggingFace - transpose!
                our_state_dict[our_key] = hf_param.t().contiguous()
            else:
                # Embeddings, LayerNorm, and biases don't need transposition
                our_state_dict[our_key] = hf_param.clone()

        # Load state dict into our model
        # strict=False because:
        # 1. lm_head.weight is tied with wte
        # 2. attn.bias is a buffer (causal mask), not a parameter in HF
        missing_keys, unexpected_keys = model.load_state_dict(our_state_dict, strict=False)

        # Filter out expected missing keys
        expected_missing = ['lm_head.weight']  # Tied with wte
        expected_missing += [f'h.{i}.attn.bias' for i in range(config.n_layer)]  # Buffers

        unexpected_missing = [k for k in missing_keys if k not in expected_missing]

        if unexpected_missing:
            print(f"Warning: Unexpected missing keys: {unexpected_missing}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        # Set to evaluation mode
        model.eval()

        print(f"✓ Successfully loaded {model_name}")
        print(f"  Total parameters: {model.get_num_params():,}")

        return model

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
            "lm_head": 0,  # Weights are tied with wte
            "total": self.get_num_params(),
        }
