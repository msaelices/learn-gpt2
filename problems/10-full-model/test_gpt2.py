"""Tests for Problem 10: Full GPT-2 Model Assembly."""

import sys
sys.path.append("../09-gpt2-config")

import pytest
import torch
from solution import GPT2Model, GPT2Config


def test_model_initialization():
    """Test that GPT2Model initializes without errors."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    assert model is not None


def test_model_has_required_components():
    """Test that model has all required components."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check embeddings
    assert hasattr(model, "wte"), "Model should have token embeddings (wte)"
    assert hasattr(model, "wpe"), "Model should have position embeddings (wpe)"
    assert hasattr(model, "drop"), "Model should have embedding dropout"

    # Check transformer blocks
    assert hasattr(model, "h"), "Model should have transformer blocks (h)"
    assert len(model.h) == config.n_layer, f"Expected {config.n_layer} blocks"

    # Check final components
    assert hasattr(model, "ln_f"), "Model should have final layer norm"
    assert hasattr(model, "lm_head"), "Model should have language model head"


def test_forward_shape():
    """Test that forward pass produces correct output shape."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)

    assert logits.shape == (
        batch_size,
        seq_len,
        config.vocab_size,
    ), f"Expected shape ({batch_size}, {seq_len}, {config.vocab_size}), got {logits.shape}"


def test_forward_with_different_batch_sizes():
    """Test that model handles different batch sizes."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    for batch_size in [1, 2, 4, 8]:
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_forward_with_different_sequence_lengths():
    """Test that model handles different sequence lengths."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    batch_size = 2
    for seq_len in [1, 5, 10, 50]:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_weight_tying():
    """Test that lm_head weights are tied with token embeddings."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Check that weights are the same object (not just equal values)
    assert (
        model.lm_head.weight is model.wte.weight
    ), "lm_head.weight should be tied with wte.weight (same object)"

    # Verify they have the same shape
    assert model.lm_head.weight.shape == model.wte.weight.shape


def test_lm_head_no_bias():
    """Test that language model head has no bias."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    assert model.lm_head.bias is None, "lm_head should not have bias"


def test_embedding_dimensions():
    """Test that embedding layers have correct dimensions."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # Token embeddings: vocab_size × n_embd
    assert model.wte.weight.shape == (config.vocab_size, config.n_embd)

    # Position embeddings: n_positions × n_embd
    assert model.wpe.weight.shape == (config.n_positions, config.n_embd)


def test_forward_produces_different_outputs_for_different_inputs():
    """Test that different inputs produce different outputs."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    model.eval()  # Set to eval mode to disable dropout

    batch_size, seq_len = 2, 10
    input_ids1 = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    input_ids2 = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits1 = model(input_ids1)
        logits2 = model(input_ids2)

    # Unless inputs are identical, outputs should be different
    if not torch.equal(input_ids1, input_ids2):
        assert not torch.allclose(
            logits1, logits2, atol=1e-5
        ), "Different inputs should produce different outputs"


def test_gradient_flow():
    """Test that gradients flow through the entire model."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Check that embeddings have gradients
    assert model.wte.weight.grad is not None, "Token embeddings should have gradients"
    assert model.wpe.weight.grad is not None, "Position embeddings should have gradients"

    # Check that transformer blocks have gradients
    for i, block in enumerate(model.h):
        assert block.attn.c_attn.weight.grad is not None, f"Block {i} attention should have gradients"
        assert block.mlp.c_fc.weight.grad is not None, f"Block {i} FFN should have gradients"


def test_model_with_gpt2_small_config():
    """Test that model works with GPT-2 small configuration."""
    config = GPT2Config.gpt2_small()
    model = GPT2Model(config)

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_model_with_gpt2_medium_config():
    """Test that model works with GPT-2 medium configuration."""
    config = GPT2Config.gpt2_medium()
    model = GPT2Model(config)

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_position_embeddings_used():
    """Test that position embeddings are actually used."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    model.eval()

    # Create input where all tokens are the same
    batch_size, seq_len = 1, 5
    same_token = 100
    input_ids = torch.full((batch_size, seq_len), same_token)

    with torch.no_grad():
        logits = model(input_ids)

    # Even with same tokens, positions should produce different outputs
    # Check that outputs at different positions are not all identical
    position_outputs = logits[0]  # Shape: (seq_len, vocab_size)

    # Compare consecutive positions
    differences = []
    for i in range(seq_len - 1):
        diff = torch.abs(position_outputs[i] - position_outputs[i + 1]).sum().item()
        differences.append(diff)

    # At least some positions should have different outputs
    assert sum(d > 1e-5 for d in differences) > 0, "Position embeddings should affect output"


def test_config_stored():
    """Test that config is stored in the model."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    assert hasattr(model, "config"), "Model should store config"
    assert model.config is config, "Stored config should be the same object"


def test_model_deterministic_in_eval_mode():
    """Test that model produces deterministic outputs in eval mode."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)
    model.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits1 = model(input_ids)
        logits2 = model(input_ids)

    assert torch.allclose(
        logits1, logits2, atol=1e-6
    ), "Model should be deterministic in eval mode"


def test_transformer_blocks_sequential():
    """Test that transformer blocks are applied sequentially."""
    config = GPT2Config(n_embd=256, n_layer=3, n_head=4)
    model = GPT2Model(config)

    # Check that we have 3 blocks
    assert len(model.h) == 3

    # Each block should be a TransformerBlock
    from solution import TransformerBlock

    for block in model.h:
        assert isinstance(block, TransformerBlock)


def test_final_layer_norm_applied():
    """Test that final layer norm is applied correctly."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    assert isinstance(model.ln_f, torch.nn.LayerNorm)
    assert model.ln_f.normalized_shape == (config.n_embd,)
    assert model.ln_f.eps == config.layer_norm_epsilon


def test_different_vocab_sizes():
    """Test that model works with different vocabulary sizes."""
    for vocab_size in [1000, 10000, 50257]:
        config = GPT2Config(vocab_size=vocab_size, n_embd=256, n_layer=2, n_head=4)
        model = GPT2Model(config)

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, vocab_size)


def test_max_sequence_length():
    """Test that model handles maximum sequence length."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4, n_positions=512)
    model = GPT2Model(config)

    batch_size = 1
    seq_len = config.n_positions  # Maximum length
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_parameter_count_helper():
    """Test parameter count helper method if implemented."""
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2Model(config)

    # If the helper method exists, test it
    if hasattr(model, "get_num_params"):
        total_params = model.get_num_params()
        manual_count = sum(p.numel() for p in model.parameters())
        assert total_params == manual_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
