"""Problem 9: GPT-2 Configuration

Learning objectives:
- Organize model hyperparameters
- Support different model sizes
- Validate configuration constraints
- Create reusable config objects

TODO: Implement the GPT2Config class below.
"""


class GPT2Config:
    """Configuration for GPT-2 model.

    This class stores all hyperparameters needed to construct a GPT-2 model.
    It provides preset configurations for the four official GPT-2 model sizes.
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

        Hints:
            - Store all parameters as instance attributes
            - Validate that n_embd is divisible by n_head
            - Compute n_inner as 4 * n_embd
            - All parameters should be accessible as self.parameter_name
        """
        # TODO: Validate that n_embd is divisible by n_head
        # TODO: Store all configuration parameters as instance attributes
        # TODO: Compute n_inner = 4 * n_embd
        raise NotImplementedError("Complete the __init__ method")

    @classmethod
    def gpt2_small(cls) -> "GPT2Config":
        """Create configuration for GPT-2 small (124M parameters).

        Returns:
            GPT2Config: Configuration for GPT-2 small.

        Hints:
            - This is the default configuration
            - Just return cls() with no arguments
        """
        # TODO: Return default configuration
        raise NotImplementedError("Complete the gpt2_small method")

    @classmethod
    def gpt2_medium(cls) -> "GPT2Config":
        """Create configuration for GPT-2 medium (355M parameters).

        Returns:
            GPT2Config: Configuration for GPT-2 medium.

        Hints:
            - n_embd=1024, n_layer=24, n_head=16
            - Other parameters use defaults
        """
        # TODO: Return configuration with medium model parameters
        raise NotImplementedError("Complete the gpt2_medium method")

    @classmethod
    def gpt2_large(cls) -> "GPT2Config":
        """Create configuration for GPT-2 large (774M parameters).

        Returns:
            GPT2Config: Configuration for GPT-2 large.

        Hints:
            - n_embd=1280, n_layer=36, n_head=20
            - Other parameters use defaults
        """
        # TODO: Return configuration with large model parameters
        raise NotImplementedError("Complete the gpt2_large method")

    @classmethod
    def gpt2_xl(cls) -> "GPT2Config":
        """Create configuration for GPT-2 XL (1.5B parameters).

        Returns:
            GPT2Config: Configuration for GPT-2 XL.

        Hints:
            - n_embd=1600, n_layer=48, n_head=25
            - Other parameters use defaults
        """
        # TODO: Return configuration with XL model parameters
        raise NotImplementedError("Complete the gpt2_xl method")
