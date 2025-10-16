"""Tests for Problem 9: GPT-2 Configuration."""

import pytest
# uncomment the following line and comment the next one
# when you have implemented the problem
# from .problem import GPT2Config
from .solution import GPT2Config


def test_config_initialization():
    """Test that GPT2Config initializes without errors."""
    config = GPT2Config()
    assert config is not None


def test_config_default_parameters():
    """Test that default parameters match GPT-2 small."""
    config = GPT2Config()

    assert config.vocab_size == 50257
    assert config.n_positions == 1024
    assert config.n_embd == 768
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.resid_pdrop == 0.1
    assert config.embd_pdrop == 0.1
    assert config.attn_pdrop == 0.1
    assert config.layer_norm_epsilon == 1e-5


def test_config_computed_n_inner():
    """Test that n_inner is computed as 4 * n_embd."""
    config = GPT2Config(n_embd=768, n_head=12)
    assert config.n_inner == 4 * 768

    config = GPT2Config(n_embd=1024, n_head=16)
    assert config.n_inner == 4 * 1024

    config = GPT2Config(n_embd=512, n_head=8)
    assert config.n_inner == 4 * 512


def test_config_custom_parameters():
    """Test that custom parameters are stored correctly."""
    config = GPT2Config(
        vocab_size=10000,
        n_positions=512,
        n_embd=256,
        n_layer=6,
        n_head=8,
        resid_pdrop=0.2,
        embd_pdrop=0.3,
        attn_pdrop=0.4,
        layer_norm_epsilon=1e-6,
    )

    assert config.vocab_size == 10000
    assert config.n_positions == 512
    assert config.n_embd == 256
    assert config.n_layer == 6
    assert config.n_head == 8
    assert config.resid_pdrop == 0.2
    assert config.embd_pdrop == 0.3
    assert config.attn_pdrop == 0.4
    assert config.layer_norm_epsilon == 1e-6
    assert config.n_inner == 4 * 256


def test_config_validation_n_embd_divisible_by_n_head():
    """Test that n_embd must be divisible by n_head."""
    # Valid: 768 is divisible by 12
    config = GPT2Config(n_embd=768, n_head=12)
    assert config.n_embd == 768
    assert config.n_head == 12

    # Invalid: 770 is not divisible by 12
    with pytest.raises(AssertionError):
        GPT2Config(n_embd=770, n_head=12)

    # Invalid: 768 is not divisible by 11
    with pytest.raises(AssertionError):
        GPT2Config(n_embd=768, n_head=11)


def test_config_gpt2_small():
    """Test GPT-2 small preset configuration."""
    config = GPT2Config.gpt2_small()

    assert config.n_embd == 768
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_inner == 4 * 768
    assert config.vocab_size == 50257
    assert config.n_positions == 1024


def test_config_gpt2_medium():
    """Test GPT-2 medium preset configuration."""
    config = GPT2Config.gpt2_medium()

    assert config.n_embd == 1024
    assert config.n_layer == 24
    assert config.n_head == 16
    assert config.n_inner == 4 * 1024
    assert config.vocab_size == 50257
    assert config.n_positions == 1024


def test_config_gpt2_large():
    """Test GPT-2 large preset configuration."""
    config = GPT2Config.gpt2_large()

    assert config.n_embd == 1280
    assert config.n_layer == 36
    assert config.n_head == 20
    assert config.n_inner == 4 * 1280
    assert config.vocab_size == 50257
    assert config.n_positions == 1024


def test_config_gpt2_xl():
    """Test GPT-2 XL preset configuration."""
    config = GPT2Config.gpt2_xl()

    assert config.n_embd == 1600
    assert config.n_layer == 48
    assert config.n_head == 25
    assert config.n_inner == 4 * 1600
    assert config.vocab_size == 50257
    assert config.n_positions == 1024


def test_config_all_presets_valid():
    """Test that all preset configurations are valid."""
    presets = [
        GPT2Config.gpt2_small(),
        GPT2Config.gpt2_medium(),
        GPT2Config.gpt2_large(),
        GPT2Config.gpt2_xl(),
    ]

    for config in presets:
        # All should initialize without errors
        assert config is not None
        # All should have n_embd divisible by n_head
        assert config.n_embd % config.n_head == 0
        # All should have computed n_inner
        assert config.n_inner == 4 * config.n_embd


def test_config_head_dim_calculation():
    """Test that head dimension is correctly divisible."""
    configs = [
        GPT2Config.gpt2_small(),
        GPT2Config.gpt2_medium(),
        GPT2Config.gpt2_large(),
        GPT2Config.gpt2_xl(),
    ]

    for config in configs:
        head_dim = config.n_embd // config.n_head
        assert head_dim > 0
        assert head_dim * config.n_head == config.n_embd


def test_config_preset_parameters():
    """Test specific parameter values for each preset."""
    # Small (124M params)
    small = GPT2Config.gpt2_small()
    assert (small.n_embd, small.n_layer, small.n_head) == (768, 12, 12)

    # Medium (355M params)
    medium = GPT2Config.gpt2_medium()
    assert (medium.n_embd, medium.n_layer, medium.n_head) == (1024, 24, 16)

    # Large (774M params)
    large = GPT2Config.gpt2_large()
    assert (large.n_embd, large.n_layer, large.n_head) == (1280, 36, 20)

    # XL (1.5B params)
    xl = GPT2Config.gpt2_xl()
    assert (xl.n_embd, xl.n_layer, xl.n_head) == (1600, 48, 25)


def test_config_dropout_probabilities():
    """Test that dropout probabilities are stored correctly."""
    config = GPT2Config()
    assert 0 <= config.resid_pdrop <= 1
    assert 0 <= config.embd_pdrop <= 1
    assert 0 <= config.attn_pdrop <= 1

    config = GPT2Config(resid_pdrop=0.5, embd_pdrop=0.2, attn_pdrop=0.3)
    assert config.resid_pdrop == 0.5
    assert config.embd_pdrop == 0.2
    assert config.attn_pdrop == 0.3


def test_config_layer_norm_epsilon():
    """Test that layer_norm_epsilon is stored correctly."""
    config = GPT2Config()
    assert config.layer_norm_epsilon == 1e-5

    config = GPT2Config(layer_norm_epsilon=1e-6)
    assert config.layer_norm_epsilon == 1e-6


def test_config_preset_returns_new_instance():
    """Test that each preset call returns a new instance."""
    config1 = GPT2Config.gpt2_small()
    config2 = GPT2Config.gpt2_small()

    # Different instances
    assert config1 is not config2

    # Same values
    assert config1.n_embd == config2.n_embd
    assert config1.n_layer == config2.n_layer
    assert config1.n_head == config2.n_head


def test_config_zero_dropout():
    """Test configuration with zero dropout (for inference)."""
    config = GPT2Config(resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)

    assert config.resid_pdrop == 0.0
    assert config.embd_pdrop == 0.0
    assert config.attn_pdrop == 0.0


def test_config_different_vocab_size():
    """Test that vocab_size can be customized."""
    config = GPT2Config(vocab_size=30000)
    assert config.vocab_size == 30000

    config = GPT2Config(vocab_size=100000)
    assert config.vocab_size == 100000


def test_config_different_context_length():
    """Test that n_positions (context length) can be customized."""
    config = GPT2Config(n_positions=512)
    assert config.n_positions == 512

    config = GPT2Config(n_positions=2048)
    assert config.n_positions == 2048


def test_config_scaling_pattern():
    """Test that model size increases follow expected patterns."""
    small = GPT2Config.gpt2_small()
    medium = GPT2Config.gpt2_medium()
    large = GPT2Config.gpt2_large()
    xl = GPT2Config.gpt2_xl()

    # Layers increase
    assert small.n_layer < medium.n_layer < large.n_layer < xl.n_layer

    # Embedding dimension increases
    assert small.n_embd < medium.n_embd < large.n_embd < xl.n_embd

    # Heads increase
    assert small.n_head < medium.n_head < large.n_head < xl.n_head

    # Inner dimension increases (proportional to n_embd)
    assert small.n_inner < medium.n_inner < large.n_inner < xl.n_inner


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
