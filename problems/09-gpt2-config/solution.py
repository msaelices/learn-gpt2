"""Problem 9: GPT-2 Configuration - Solution

This solution implements a configuration class for organizing GPT-2 hyperparameters.
"""


class GPT2Config:
    """Configuration for GPT-2 model.

    This class stores all hyperparameters needed to construct a GPT-2 model.
    It validates constraints and provides preset configurations for the four
    official GPT-2 model sizes (small, medium, large, xl).
    """

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
            vocab_size: Vocabulary size (number of tokens).
            n_positions: Maximum sequence length.
            n_embd: Embedding dimension.
            n_layer: Number of transformer layers.
            n_head: Number of attention heads.
            resid_pdrop: Residual dropout probability.
            embd_pdrop: Embedding dropout probability.
            attn_pdrop: Attention dropout probability.
            layer_norm_epsilon: Layer normalization epsilon for numerical stability.

        Raises:
            AssertionError: If n_embd is not divisible by n_head.
        """
        # Validate that n_embd is divisible by n_head
        # This ensures each head gets equal dimensions
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"

        # Architecture parameters
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        # Regularization parameters
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop

        # Normalization parameters
        self.layer_norm_epsilon = layer_norm_epsilon

        # Computed attributes
        # Inner dimension of feedforward network (always 4x embedding dimension)
        self.n_inner = 4 * n_embd

    @classmethod
    def gpt2_small(cls) -> "GPT2Config":
        """Create configuration for GPT-2 small (124M parameters).

        Returns:
            GPT2Config: Configuration with default parameters (small model).

        Model specifications:
            - Layers: 12
            - Embedding dim: 768
            - Heads: 12
            - Parameters: ~124M
        """
        return cls()

    @classmethod
    def gpt2_medium(cls) -> "GPT2Config":
        """Create configuration for GPT-2 medium (355M parameters).

        Returns:
            GPT2Config: Configuration for medium model.

        Model specifications:
            - Layers: 24
            - Embedding dim: 1024
            - Heads: 16
            - Parameters: ~355M
        """
        return cls(n_embd=1024, n_layer=24, n_head=16)

    @classmethod
    def gpt2_large(cls) -> "GPT2Config":
        """Create configuration for GPT-2 large (774M parameters).

        Returns:
            GPT2Config: Configuration for large model.

        Model specifications:
            - Layers: 36
            - Embedding dim: 1280
            - Heads: 20
            - Parameters: ~774M
        """
        return cls(n_embd=1280, n_layer=36, n_head=20)

    @classmethod
    def gpt2_xl(cls) -> "GPT2Config":
        """Create configuration for GPT-2 XL (1.5B parameters).

        Returns:
            GPT2Config: Configuration for XL model.

        Model specifications:
            - Layers: 48
            - Embedding dim: 1600
            - Heads: 25
            - Parameters: ~1.5B
        """
        return cls(n_embd=1600, n_layer=48, n_head=25)
