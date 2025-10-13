"""Tests for GPT2Model.from_pretrained method."""

import torch

from gpt2 import GPT2Model


def test_from_pretrained_loads():
    """Test that from_pretrained loads successfully."""
    model = GPT2Model.from_pretrained("gpt2")
    assert model is not None
    assert isinstance(model, GPT2Model)


def test_from_pretrained_config():
    """Test that configuration matches HF model."""
    model = GPT2Model.from_pretrained("gpt2")
    assert model.config.vocab_size == 50257
    assert model.config.n_embd == 768
    assert model.config.n_layer == 12
    assert model.config.n_head == 12
    assert model.config.n_positions == 1024


def test_from_pretrained_forward():
    """Test that forward pass works with loaded weights."""
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    input_ids = torch.randint(0, 50257, (1, 10))
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 10, 50257)


def test_from_pretrained_matches_hf():
    """Test that output matches HF model (within tolerance)."""
    from transformers import GPT2LMHeadModel

    # Load both models
    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set to eval mode
    our_model.eval()
    hf_model.eval()

    # Test with same input
    torch.manual_seed(42)
    input_ids = torch.randint(0, 50257, (2, 10))

    with torch.no_grad():
        our_logits = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    # Check shapes match
    assert our_logits.shape == hf_logits.shape

    # Check values are close (allowing for numerical differences)
    max_diff = torch.max(torch.abs(our_logits - hf_logits)).item()
    print(f"Maximum difference between our model and HF model: {max_diff}")

    assert torch.allclose(our_logits, hf_logits, atol=1e-4, rtol=1e-3), (
        f"Model outputs differ too much. Max diff: {max_diff}"
    )


def test_from_pretrained_deterministic():
    """Test that the same input produces the same output."""
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        output1 = model(input_ids)
        output2 = model(input_ids)

    assert torch.allclose(output1, output2), "Model outputs are not deterministic"


def test_from_pretrained_weight_tying():
    """Test that embeddings and lm_head weights are tied."""
    model = GPT2Model.from_pretrained("gpt2")

    # Check that wte and lm_head share the same weight tensor
    assert model.wte.weight is model.lm_head.weight, (
        "Token embeddings and lm_head weights should be tied"
    )
