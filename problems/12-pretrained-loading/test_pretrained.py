"""Tests for Problem 12: Loading Pretrained Weights."""

import sys

sys.path.append("../09-gpt2-config")

import pytest
import torch
# uncomment the following line and comment the next one
# when you have implemented the problem
# from .problem import GPT2Model
from .solution import GPT2Model


def test_from_pretrained_exists():
    """Test that from_pretrained method exists."""
    assert hasattr(GPT2Model, "from_pretrained"), "GPT2Model should have from_pretrained method"


def test_from_pretrained_loads_model():
    """Test that from_pretrained loads a model without errors."""
    model = GPT2Model.from_pretrained("gpt2")
    assert model is not None
    assert isinstance(model, GPT2Model)


def test_loaded_model_has_correct_config():
    """Test that loaded model has correct configuration."""
    model = GPT2Model.from_pretrained("gpt2")

    # GPT-2 small specs
    assert model.config.n_layer == 12
    assert model.config.n_embd == 768
    assert model.config.n_head == 12
    assert model.config.vocab_size == 50257
    assert model.config.n_positions == 1024


def test_forward_pass_works():
    """Test that forward pass works with loaded model."""
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, model.config.vocab_size)


def test_output_matches_huggingface():
    """Test that our output matches HuggingFace's output."""
    from transformers import GPT2LMHeadModel

    # Load both models
    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    our_model.eval()
    hf_model.eval()

    # Test with same input
    input_ids = torch.tensor([[15496, 995, 318]])  # "Hello World is"

    with torch.no_grad():
        our_logits = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    # Outputs should match within tolerance
    assert torch.allclose(our_logits, hf_logits, atol=1e-5), (
        f"Max diff: {(our_logits - hf_logits).abs().max().item()}"
    )


def test_output_matches_on_longer_sequence():
    """Test that output matches HuggingFace on longer sequence."""
    from transformers import GPT2LMHeadModel

    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    our_model.eval()
    hf_model.eval()

    # Longer sequence
    input_ids = torch.randint(0, 50257, (1, 50))

    with torch.no_grad():
        our_logits = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    assert torch.allclose(our_logits, hf_logits, atol=1e-5)


def test_output_matches_on_batch():
    """Test that output matches HuggingFace on batched input."""
    from transformers import GPT2LMHeadModel

    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    our_model.eval()
    hf_model.eval()

    # Batch of inputs
    input_ids = torch.randint(0, 50257, (4, 20))

    with torch.no_grad():
        our_logits = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    assert torch.allclose(our_logits, hf_logits, atol=1e-5)


def test_weight_tying_preserved():
    """Test that weight tying is preserved after loading."""
    model = GPT2Model.from_pretrained("gpt2")

    # Weight tying should still be in place
    assert model.lm_head.weight is model.wte.weight, (
        "Weight tying should be preserved after loading"
    )


def test_parameter_count_matches():
    """Test that parameter count matches expected."""
    model = GPT2Model.from_pretrained("gpt2")

    # GPT-2 small has approximately 124M parameters
    total_params = model.get_num_params()
    expected_params = 124439808  # Exact count for GPT-2 small

    assert total_params == expected_params, (
        f"Expected {expected_params:,} parameters, got {total_params:,}"
    )


def test_all_weights_loaded():
    """Test that all weights are loaded (no NaN or random values)."""
    model = GPT2Model.from_pretrained("gpt2")

    # Check that no parameters are NaN
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"Parameter {name} contains NaN"

    # Check that weights are not at initialization values
    # (should be different from random init)
    linear_layer = model.h[0].attn.c_attn
    mean = linear_layer.weight.mean().item()
    std = linear_layer.weight.std().item()

    # Pretrained weights will have different statistics than initialization
    # (initialization is mean=0, std=0.02)
    # Pretrained weights typically have slightly different statistics
    assert abs(mean) < 0.5  # Should be relatively small
    assert 0.01 < std < 0.5  # Should be in reasonable range (pretrained can be larger)


def test_model_in_eval_mode():
    """Test that loaded model is in eval mode."""
    model = GPT2Model.from_pretrained("gpt2")
    assert not model.training, "Loaded model should be in eval mode"


def test_deterministic_output():
    """Test that model produces deterministic output in eval mode."""
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    input_ids = torch.tensor([[15496, 995]])

    with torch.no_grad():
        logits1 = model(input_ids)
        logits2 = model(input_ids)

    assert torch.equal(logits1, logits2), "Model should be deterministic in eval mode"


def test_specific_token_predictions():
    """Test specific token predictions match HuggingFace."""
    from transformers import GPT2LMHeadModel

    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    our_model.eval()
    hf_model.eval()

    # "The quick brown"
    input_ids = torch.tensor([[464, 2068, 7586]])

    with torch.no_grad():
        our_logits = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    # Get top-5 predictions from both models
    our_top5 = torch.topk(our_logits[0, -1], 5).indices
    hf_top5 = torch.topk(hf_logits[0, -1], 5).indices

    # Top predictions should be identical
    assert torch.equal(our_top5, hf_top5), (
        f"Top-5 predictions differ: ours={our_top5.tolist()}, hf={hf_top5.tolist()}"
    )


def test_embedding_weights_match():
    """Test that embedding weights match HuggingFace exactly."""
    from transformers import GPT2LMHeadModel

    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    our_wte = our_model.wte.weight
    hf_wte = hf_model.transformer.wte.weight

    assert torch.allclose(our_wte, hf_wte, atol=1e-7), "Token embeddings should match exactly"

    our_wpe = our_model.wpe.weight
    hf_wpe = hf_model.transformer.wpe.weight

    assert torch.allclose(our_wpe, hf_wpe, atol=1e-7), "Position embeddings should match exactly"


def test_attention_weights_match():
    """Test that attention weights match HuggingFace (after transposition)."""
    from transformers import GPT2LMHeadModel

    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Check first layer attention weights
    our_attn = our_model.h[0].attn.c_attn.weight
    hf_attn = hf_model.transformer.h[0].attn.c_attn.weight

    # Our weights should match HuggingFace's transposed weights
    assert torch.allclose(our_attn, hf_attn.t(), atol=1e-7), (
        "Attention weights should match (after transposition)"
    )


def test_ffn_weights_match():
    """Test that FFN weights match HuggingFace (after transposition)."""
    from transformers import GPT2LMHeadModel

    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Check first layer FFN weights
    our_fc = our_model.h[0].mlp.c_fc.weight
    hf_fc = hf_model.transformer.h[0].mlp.c_fc.weight

    # Our weights should match HuggingFace's transposed weights
    assert torch.allclose(our_fc, hf_fc.t(), atol=1e-7), (
        "FFN weights should match (after transposition)"
    )


def test_layernorm_weights_match():
    """Test that LayerNorm weights match HuggingFace exactly."""
    from transformers import GPT2LMHeadModel

    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Check first layer norm
    our_ln1 = our_model.h[0].ln_1.weight
    hf_ln1 = hf_model.transformer.h[0].ln_1.weight

    assert torch.allclose(our_ln1, hf_ln1, atol=1e-7), "LayerNorm weights should match exactly"

    # Check final layer norm
    our_ln_f = our_model.ln_f.weight
    hf_ln_f = hf_model.transformer.ln_f.weight

    assert torch.allclose(our_ln_f, hf_ln_f, atol=1e-7), (
        "Final LayerNorm weights should match exactly"
    )


def test_loading_different_sizes():
    """Test that loading works for different model sizes."""
    # Only test sizes that are reasonable for testing
    # (gpt2-medium takes ~1.4GB, gpt2-large ~3GB, gpt2-xl ~6GB)

    # Test gpt2 small
    model_small = GPT2Model.from_pretrained("gpt2")
    assert model_small.config.n_layer == 12
    assert model_small.config.n_embd == 768

    # You can add tests for other sizes if you have the memory:
    # model_medium = GPT2Model.from_pretrained("gpt2-medium")
    # assert model_medium.config.n_layer == 24
    # assert model_medium.config.n_embd == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
