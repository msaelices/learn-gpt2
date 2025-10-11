"""GPT-2 model implementation."""

import torch
import torch.nn as nn
from torch import Tensor


class GPT2Config:
    """Configuration for GPT-2 model."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        """Initialize GPT-2 configuration.

        Args:
            vocab_size: Vocabulary size.
            n_positions: Maximum sequence length.
            n_embd: Embedding dimension.
            n_layer: Number of transformer layers.
            n_head: Number of attention heads.
            n_inner: Inner dimension of feedforward network. If None, defaults to 4 * n_embd.
            activation_function: Activation function name.
            resid_pdrop: Residual dropout probability.
            embd_pdrop: Embedding dropout probability.
            attn_pdrop: Attention dropout probability.
            layer_norm_epsilon: Layer normalization epsilon.
        """
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon

        # Computed attributes

        # Inner dimension of feedforward network, 4 * n_embd.
        self.n_inner = 4 * n_embd


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, config: GPT2Config) -> None:
        """Initialize multi-head attention.

        Args:
            config: GPT-2 configuration.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # TODO: Implement

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """

        # TODO: Implement
        out = x  # Placeholder
        return out


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, config: GPT2Config) -> None:
        """Initialize feedforward network.

        Args:
            config: GPT-2 configuration.
        """
        super().__init__()
        # TODO: Configure the feedforward network properly

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        # TODO: Implement the feedforward network
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward layers."""

    def __init__(self, config: GPT2Config) -> None:
        """Initialize transformer block.

        Args:
            config: GPT-2 configuration.
        """
        super().__init__()
        # TODO: Implement inspecting the GPT2 model from Hugging Face to get the correct implementation

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        # TODO: Implement the forward pass of the transformer block
        return x


class GPT2Model(nn.Module):
    """GPT-2 language model."""

    def __init__(self, config: GPT2Config) -> None:
        """Initialize GPT-2 model.

        Args:
            config: GPT-2 configuration.
        """
        super().__init__()
        self.config = config

        # TODO: Implement inspecting the GPT2 model from Hugging Face to get the correct implementation

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        # TODO: Implement the forward pass of the GPT-2 model
        logits = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.config.vocab_size
        )  # Placeholder
        return logits
