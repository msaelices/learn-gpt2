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

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Combined Q, K, V projection (more efficient than separate projections)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask to ensure attention only flows to the left
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        batch_size, seq_len, n_embd = x.size()

        # Calculate Q, K, V for all heads in batch
        qkv = self.c_attn(x)  # (batch_size, seq_len, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to split into multiple heads
        # (batch_size, n_head, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # (batch_size, n_head, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask (prevent attending to future positions)
        attn_scores = attn_scores.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )

        # Apply optional attention mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (batch_size, n_head, seq_len, head_dim)
        out = attn_weights @ v

        # Concatenate heads and reshape
        # (batch_size, seq_len, n_embd)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        # Output projection and dropout
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, config: GPT2Config) -> None:
        """Initialize feedforward network.

        Args:
            config: GPT-2 configuration.
        """
        super().__init__()
        # Two-layer MLP with GELU activation
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward layers."""

    def __init__(self, config: GPT2Config) -> None:
        """Initialize transformer block.

        Args:
            config: GPT-2 configuration.
        """
        super().__init__()
        # Layer normalization before attention (pre-norm architecture)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # Multi-head self-attention
        self.attn = MultiHeadAttention(config)
        # Layer normalization before feedforward
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # Feedforward network
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd).
        """
        # Attention block with residual connection
        x = x + self.attn(self.ln_1(x), mask)
        # Feedforward block with residual connection
        x = x + self.mlp(self.ln_2(x))
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

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # Embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Language model head (projects to vocabulary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between token embeddings and output projection
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the model.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.size()

        # Check sequence length
        assert seq_len <= self.config.n_positions, (
            f"Sequence length {seq_len} exceeds maximum "
            f"position embeddings {self.config.n_positions}"
        )

        # Create position indices
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Get embeddings
        token_embeddings = self.wte(input_ids)  # (batch_size, seq_len, n_embd)
        position_embeddings = self.wpe(position_ids)  # (batch_size, seq_len, n_embd)

        # Combine embeddings and apply dropout
        x = token_embeddings + position_embeddings
        x = self.drop(x)

        # Pass through transformer blocks
        for block in self.h:
            x = block(x, attention_mask)

        # Apply final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits
