"""Tests for Problem 11: Weight Initialization."""

import sys

sys.path.append("../09-gpt2-config")

import pytest
import torch
import torch.nn as nn
from solution import GPT2Config, GPT2Model


def test_model_initialization():
    """Test that model initializes without errors."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    assert model is not None


def test_init_weights_method_exists():
    """Test that _init_weights method exists."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    assert hasattr(model, "_init_weights"), "Model should have _init_weights method"


def test_linear_weight_initialization():
    """Test that linear layer weights are initialized with normal(0, 0.02)."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check a linear layer from attention
    linear_layer = model.h[0].attn.c_attn

    # Check mean is close to 0
    mean = linear_layer.weight.mean().item()
    assert abs(mean) < 0.01, f"Linear weight mean should be ≈0, got {mean}"

    # Check std is close to 0.02
    std = linear_layer.weight.std().item()
    assert 0.015 < std < 0.025, f"Linear weight std should be ≈0.02, got {std}"


def test_linear_bias_initialization():
    """Test that linear layer biases are initialized to zero."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check linear layers with biases
    linear_layer = model.h[0].attn.c_attn

    if linear_layer.bias is not None:
        bias_sum = linear_layer.bias.abs().sum().item()
        assert bias_sum == 0.0, f"Linear bias should be all zeros, got sum {bias_sum}"


def test_embedding_weight_initialization():
    """Test that embedding weights are initialized with normal(0, 0.02)."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check token embeddings
    wte_mean = model.wte.weight.mean().item()
    assert abs(wte_mean) < 0.01, f"Token embedding mean should be ≈0, got {wte_mean}"

    wte_std = model.wte.weight.std().item()
    assert 0.015 < wte_std < 0.025, f"Token embedding std should be ≈0.02, got {wte_std}"

    # Check position embeddings
    wpe_mean = model.wpe.weight.mean().item()
    assert abs(wpe_mean) < 0.01, f"Position embedding mean should be ≈0, got {wpe_mean}"

    wpe_std = model.wpe.weight.std().item()
    assert 0.015 < wpe_std < 0.025, f"Position embedding std should be ≈0.02, got {wpe_std}"


def test_layernorm_weight_initialization():
    """Test that LayerNorm weights are initialized to ones."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check LayerNorm in first block
    ln_1 = model.h[0].ln_1

    # All weights should be exactly 1.0
    assert torch.all(ln_1.weight == 1.0), "LayerNorm weights should all be 1.0"


def test_layernorm_bias_initialization():
    """Test that LayerNorm biases are initialized to zeros."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check LayerNorm in first block
    ln_1 = model.h[0].ln_1

    # All biases should be exactly 0.0
    assert torch.all(ln_1.bias == 0.0), "LayerNorm biases should all be 0.0"


def test_final_layernorm_initialization():
    """Test that final LayerNorm is initialized correctly."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check final layer norm
    assert torch.all(model.ln_f.weight == 1.0), "Final LayerNorm weights should be 1.0"
    assert torch.all(model.ln_f.bias == 0.0), "Final LayerNorm biases should be 0.0"


def test_all_linear_layers_initialized():
    """Test that all linear layers in the model are initialized."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Collect all linear layers
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    # Should have multiple linear layers
    assert len(linear_layers) > 0, "Model should have linear layers"

    # Check each linear layer
    for layer in linear_layers:
        mean = layer.weight.mean().item()
        std = layer.weight.std().item()

        # Mean should be close to 0
        assert abs(mean) < 0.05, f"Linear layer mean too far from 0: {mean}"

        # Std should be reasonable (allowing some variance due to small samples)
        assert 0.01 < std < 0.04, f"Linear layer std should be ≈0.02, got {std}"


def test_all_embeddings_initialized():
    """Test that all embedding layers are initialized."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Collect all embedding layers
    embedding_layers = [m for m in model.modules() if isinstance(m, nn.Embedding)]

    # Should have 2 embedding layers (token + position)
    assert len(embedding_layers) == 2, f"Expected 2 embeddings, got {len(embedding_layers)}"

    # Check each embedding layer
    for layer in embedding_layers:
        mean = layer.weight.mean().item()
        std = layer.weight.std().item()

        # Mean should be close to 0
        assert abs(mean) < 0.05, f"Embedding mean too far from 0: {mean}"

        # Std should be reasonable
        assert 0.01 < std < 0.04, f"Embedding std should be ≈0.02, got {std}"


def test_all_layernorms_initialized():
    """Test that all LayerNorm layers are initialized."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Collect all LayerNorm layers
    ln_layers = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]

    # Should have multiple LayerNorm layers
    # 2 per transformer block + 1 final = 2*2 + 1 = 5
    assert len(ln_layers) >= 3, "Model should have LayerNorm layers"

    # Check each LayerNorm
    for layer in ln_layers:
        assert torch.all(layer.weight == 1.0), "All LayerNorm weights should be 1.0"
        assert torch.all(layer.bias == 0.0), "All LayerNorm biases should be 0.0"


def test_no_nan_or_inf_after_initialization():
    """Test that initialization doesn't produce NaN or inf values."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check all parameters
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"Parameter {name} contains NaN"
        assert not torch.isinf(param).any(), f"Parameter {name} contains inf"


def test_forward_pass_after_initialization():
    """Test that forward pass works correctly after initialization."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    model.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # Check for NaN or inf in output
    assert not torch.isnan(logits).any(), "Output contains NaN"
    assert not torch.isinf(logits).any(), "Output contains inf"


def test_activations_reasonable_scale():
    """Test that activations after initialization are at a reasonable scale."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    model.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    # Check that logits are at a reasonable scale (not too large or too small)
    logit_mean = logits.abs().mean().item()
    assert logit_mean < 100, f"Logits too large: {logit_mean}"
    assert logit_mean > 0.01, f"Logits too small: {logit_mean}"


def test_gradient_flow_after_initialization():
    """Test that gradients flow properly through initialized model."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Check that gradients exist and are reasonable
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
            assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains inf"


def test_different_seeds_produce_different_weights():
    """Test that different random seeds produce different initializations."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)

    # Create first model with seed 42
    torch.manual_seed(42)
    model1 = GPT2Model(config)
    weights1 = model1.h[0].attn.c_attn.weight.clone()

    # Create second model with seed 123
    torch.manual_seed(123)
    model2 = GPT2Model(config)
    weights2 = model2.h[0].attn.c_attn.weight.clone()

    # Weights should be different
    assert not torch.allclose(weights1, weights2), "Different seeds should produce different weights"


def test_same_seed_produces_same_weights():
    """Test that same random seed produces identical initialization."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)

    # Create first model with seed 42
    torch.manual_seed(42)
    model1 = GPT2Model(config)
    weights1 = model1.h[0].attn.c_attn.weight.clone()

    # Create second model with same seed
    torch.manual_seed(42)
    model2 = GPT2Model(config)
    weights2 = model2.h[0].attn.c_attn.weight.clone()

    # Weights should be identical
    assert torch.allclose(weights1, weights2), "Same seed should produce identical weights"


def test_weight_tying_preserved():
    """Test that weight tying is preserved after initialization."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Weight tying should still be in place
    assert model.lm_head.weight is model.wte.weight, "Weight tying should be preserved"


def test_initialization_with_different_model_sizes():
    """Test that initialization works for different model sizes."""
    configs = [
        GPT2Config.gpt2_small(),
        GPT2Config(n_embd=512, n_layer=4, n_head=8),
    ]

    for config in configs:
        model = GPT2Model(config)

        # Check token embeddings
        mean = model.wte.weight.mean().item()
        std = model.wte.weight.std().item()

        assert abs(mean) < 0.05, f"Mean should be ≈0 for config, got {mean}"
        assert 0.01 < std < 0.04, f"Std should be ≈0.02 for config, got {std}"


def test_lm_head_no_bias():
    """Test that lm_head has no bias (as expected)."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # lm_head should not have bias
    assert model.lm_head.bias is None, "lm_head should not have bias"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
