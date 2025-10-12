"""Tests for Problem 1: Token & Position Embeddings

Run with: python -m pytest test_embeddings.py -v
"""

import pytest
import torch
import torch.nn as nn
from problem import Embeddings


def test_initialization():
    """Test that Embeddings module initializes without errors."""
    embeddings = Embeddings(
        vocab_size=50257, n_positions=1024, n_embd=768, embd_pdrop=0.1
    )
    assert embeddings is not None
    assert isinstance(embeddings, nn.Module)


def test_has_required_layers():
    """Test that module has token embeddings, position embeddings, and dropout."""
    embeddings = Embeddings()

    # Check for token embedding layer
    assert hasattr(embeddings, "wte"), "Missing token embedding layer (wte)"
    assert isinstance(
        embeddings.wte, nn.Embedding
    ), "wte should be an nn.Embedding layer"

    # Check for position embedding layer
    assert hasattr(embeddings, "wpe"), "Missing position embedding layer (wpe)"
    assert isinstance(
        embeddings.wpe, nn.Embedding
    ), "wpe should be an nn.Embedding layer"

    # Check for dropout layer
    assert hasattr(embeddings, "drop"), "Missing dropout layer (drop)"
    assert isinstance(embeddings.drop, nn.Dropout), "drop should be an nn.Dropout layer"


def test_embedding_dimensions():
    """Test that embedding layers have correct dimensions."""
    vocab_size = 1000
    n_positions = 512
    n_embd = 256

    embeddings = Embeddings(
        vocab_size=vocab_size, n_positions=n_positions, n_embd=n_embd
    )

    # Check token embedding dimensions
    assert (
        embeddings.wte.num_embeddings == vocab_size
    ), f"Token embedding should have {vocab_size} embeddings"
    assert (
        embeddings.wte.embedding_dim == n_embd
    ), f"Token embedding dimension should be {n_embd}"

    # Check position embedding dimensions
    assert (
        embeddings.wpe.num_embeddings == n_positions
    ), f"Position embedding should have {n_positions} embeddings"
    assert (
        embeddings.wpe.embedding_dim == n_embd
    ), f"Position embedding dimension should be {n_embd}"


def test_forward_output_shape():
    """Test that forward pass produces correct output shape."""
    batch_size = 4
    seq_len = 20
    n_embd = 768

    embeddings = Embeddings(vocab_size=50257, n_positions=1024, n_embd=n_embd)

    # Create random input token IDs
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))

    # Forward pass
    output = embeddings(input_ids)

    # Check output shape
    expected_shape = (batch_size, seq_len, n_embd)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"


def test_different_sequence_lengths():
    """Test that module works with different sequence lengths."""
    embeddings = Embeddings(vocab_size=50257, n_positions=1024, n_embd=768)

    for seq_len in [1, 10, 50, 100, 512]:
        input_ids = torch.randint(0, 50257, (2, seq_len))
        output = embeddings(input_ids)
        assert output.shape == (
            2,
            seq_len,
            768,
        ), f"Failed for sequence length {seq_len}"


def test_different_batch_sizes():
    """Test that module works with different batch sizes."""
    embeddings = Embeddings(vocab_size=50257, n_positions=1024, n_embd=768)

    for batch_size in [1, 2, 8, 16]:
        input_ids = torch.randint(0, 50257, (batch_size, 20))
        output = embeddings(input_ids)
        assert output.shape == (
            batch_size,
            20,
            768,
        ), f"Failed for batch size {batch_size}"


def test_different_tokens_produce_different_embeddings():
    """Test that different input tokens produce different embeddings."""
    embeddings = Embeddings()
    embeddings.eval()  # Set to eval mode to disable dropout

    # Create two different inputs
    input_ids_1 = torch.tensor([[1, 2, 3, 4, 5]])
    input_ids_2 = torch.tensor([[5, 4, 3, 2, 1]])

    with torch.no_grad():
        output_1 = embeddings(input_ids_1)
        output_2 = embeddings(input_ids_2)

    # Outputs should be different
    assert not torch.allclose(
        output_1, output_2
    ), "Different inputs should produce different embeddings"


def test_position_information_added():
    """Test that position embeddings are being added."""
    embeddings = Embeddings()
    embeddings.eval()  # Disable dropout

    # Same tokens at different positions should have different embeddings
    input_ids = torch.tensor([[42, 42, 42, 42, 42]])

    with torch.no_grad():
        output = embeddings(input_ids)

    # Each position should have a different embedding (due to position embeddings)
    for i in range(4):
        assert not torch.allclose(
            output[0, i], output[0, i + 1]
        ), f"Position {i} and {i+1} should have different embeddings"


def test_dropout_behavior():
    """Test that dropout is applied in training mode but not in eval mode."""
    embeddings = Embeddings(embd_pdrop=0.5)  # High dropout for testing
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    # Training mode: outputs should vary due to dropout
    embeddings.train()
    outputs_train = []
    for _ in range(5):
        output = embeddings(input_ids)
        outputs_train.append(output.clone())

    # At least some outputs should be different (due to dropout randomness)
    all_same = all(
        torch.allclose(outputs_train[0], out, atol=1e-6) for out in outputs_train[1:]
    )
    assert not all_same, "Dropout should cause variation in training mode"

    # Eval mode: outputs should be identical
    embeddings.eval()
    with torch.no_grad():
        output_eval_1 = embeddings(input_ids)
        output_eval_2 = embeddings(input_ids)

    assert torch.allclose(
        output_eval_1, output_eval_2, atol=1e-6
    ), "Outputs should be identical in eval mode (no dropout)"


def test_device_compatibility():
    """Test that module works on different devices."""
    embeddings = Embeddings()

    # Test on CPU
    input_ids_cpu = torch.randint(0, 50257, (2, 10))
    output_cpu = embeddings(input_ids_cpu)
    assert output_cpu.device == input_ids_cpu.device, "Output should be on same device as input"

    # Test on GPU if available
    if torch.cuda.is_available():
        embeddings_gpu = embeddings.cuda()
        input_ids_gpu = torch.randint(0, 50257, (2, 10), device="cuda")
        output_gpu = embeddings_gpu(input_ids_gpu)
        assert output_gpu.device == input_ids_gpu.device, "Output should be on GPU"


def test_maximum_sequence_length():
    """Test that module handles sequences up to max length."""
    n_positions = 1024
    embeddings = Embeddings(n_positions=n_positions)

    # Should work with sequence length equal to n_positions
    input_ids = torch.randint(0, 50257, (1, n_positions))
    output = embeddings(input_ids)
    assert output.shape == (1, n_positions, 768)


def test_gradients_flow():
    """Test that gradients flow through the module."""
    embeddings = Embeddings()
    input_ids = torch.randint(0, 50257, (2, 10))

    output = embeddings(input_ids)
    loss = output.sum()
    loss.backward()

    # Check that embedding layers have gradients
    assert (
        embeddings.wte.weight.grad is not None
    ), "Token embeddings should have gradients"
    assert (
        embeddings.wpe.weight.grad is not None
    ), "Position embeddings should have gradients"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
