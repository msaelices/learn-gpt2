# Problem 9: GPT-2 Configuration

## Learning Objectives
- Organize model hyperparameters in a reusable configuration class
- Support different GPT-2 model sizes (small, medium, large, xl)
- Validate configuration constraints
- Understand the relationship between parameters

## Background

As models grow in complexity, managing hyperparameters becomes crucial. A configuration class centralizes all model settings, making it easy to:
- Create different model sizes
- Share configurations between training and inference
- Validate parameter constraints
- Document model architecture

### GPT-2 Model Sizes

OpenAI released four GPT-2 model sizes:

| Model | Parameters | Layers | Embedding Dim | Heads | Context Length |
|-------|------------|--------|---------------|-------|----------------|
| **Small** | 124M | 12 | 768 | 12 | 1024 |
| **Medium** | 355M | 24 | 1024 | 16 | 1024 |
| **Large** | 774M | 36 | 1280 | 20 | 1024 |
| **XL** | 1.5B | 48 | 1600 | 25 | 1024 |

All models share:
- Vocabulary size: 50,257 tokens
- Context length: 1024 tokens
- Inner FFN dimension: 4× embedding dimension

### Key Configuration Parameters

**Architecture Parameters**:
- `vocab_size`: Number of tokens in vocabulary (50257 for GPT-2)
- `n_positions`: Maximum sequence length (1024)
- `n_embd`: Embedding dimension (varies by model size)
- `n_layer`: Number of transformer blocks (varies by model size)
- `n_head`: Number of attention heads (varies by model size)
- `n_inner`: FFN inner dimension (computed as 4 * n_embd)

**Regularization Parameters**:
- `resid_pdrop`: Dropout probability for residual connections (0.1)
- `embd_pdrop`: Dropout probability for embeddings (0.1)
- `attn_pdrop`: Dropout probability for attention weights (0.1)

**Normalization Parameters**:
- `layer_norm_epsilon`: Small constant for numerical stability in LayerNorm (1e-5)

### Configuration Validation

The configuration must validate that:
- `n_embd` is divisible by `n_head` (so each head gets equal dimensions)
- All parameters are positive
- Dropout probabilities are between 0 and 1

## Your Task

Implement the `GPT2Config` class that:
1. Stores all model hyperparameters
2. Validates configuration constraints
3. Computes derived values (like `n_inner`)
4. Provides class methods for standard model sizes

## Hints

💡 **Getting Started**
- This is a simple Python class (not a `nn.Module`)
- Store all parameters as instance attributes in `__init__`
- Use `@classmethod` decorators for model size presets

💡 **Implementation Tips**
- Validate `n_embd % n_head == 0` to ensure equal head dimensions
- Compute `n_inner = 4 * n_embd` automatically
- Use default values that match GPT-2 small
- Class methods should return `cls(...)` with appropriate parameters

💡 **Common Pitfalls**
- Don't forget to validate `n_embd` divisibility by `n_head`
- Remember to set ALL parameters, not just the ones that change
- Class methods use `cls` not `self`
- Make sure `n_inner` is computed, not passed as a parameter

💡 **Testing Tips**
- Test that all parameters are stored correctly
- Verify validation catches invalid configurations
- Check that preset configs match official GPT-2 specs
- Ensure `n_inner` is always 4 * `n_embd`

## Testing Your Solution

```bash
cd problems/09-gpt2-config
uv run pytest test_config.py -v
```

## Resources

📚 **Python Documentation**
- [Class Methods](https://docs.python.org/3/library/functions.html#classmethod) - `@classmethod` decorator
- [Dataclasses](https://docs.python.org/3/library/dataclasses.html) - Alternative approach (optional)

📄 **Papers & Articles**
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-2 Model Card](https://github.com/openai/gpt-2/blob/master/model_card.md) - Official specifications

💻 **Additional Resources**
- [HuggingFace GPT-2 Configuration](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config)
- [Python Class Methods vs Static Methods](https://realpython.com/instance-class-and-static-methods-demystified/)

## Key Concepts

**Why Configuration Classes?**
- Centralize all hyperparameters in one place
- Easy to create different model variants
- Self-documenting (parameters are explicit)
- Type checking and validation
- Reusable across training, evaluation, and deployment

**Model Scaling Patterns**:
```
Small → Medium:  n_layer ×2, n_embd ×1.33, n_head ×1.33
Medium → Large:  n_layer ×1.5, n_embd ×1.25, n_head ×1.25
Large → XL:      n_layer ×1.33, n_embd ×1.25, n_head ×1.25
```

**Why n_embd must be divisible by n_head?**
```
head_dim = n_embd // n_head

Example (GPT-2 small):
  n_embd = 768
  n_head = 12
  head_dim = 768 // 12 = 64 ✓

Bad example:
  n_embd = 770
  n_head = 12
  head_dim = 770 // 12 = 64.16... ✗ (not an integer!)
```

## Configuration Examples

**Creating Configurations**:
```python
# Default (GPT-2 small)
config = GPT2Config()

# Custom configuration
config = GPT2Config(
    n_embd=512,
    n_layer=6,
    n_head=8,
)

# Preset configurations
config_small = GPT2Config.gpt2_small()
config_medium = GPT2Config.gpt2_medium()
config_large = GPT2Config.gpt2_large()
config_xl = GPT2Config.gpt2_xl()
```

**Using Configurations**:
```python
# Pass to model components
config = GPT2Config()

# MultiHeadAttention will use config.n_embd, config.n_head, etc.
attention = MultiHeadAttention(config)

# FeedForward will use config.n_embd, config.n_inner
feedforward = FeedForward(config)

# TransformerBlock will use the full config
block = TransformerBlock(config)
```

## Parameter Count Formula

Total parameters ≈ (embeddings) + (layers × params_per_layer) + (lm_head)

```
vocab_params = vocab_size × n_embd        # Token embeddings
pos_params = n_positions × n_embd         # Position embeddings

attn_params = 4 × n_embd²                 # Q,K,V projection + output
ffn_params = 2 × n_embd × n_inner         # Two linear layers
ln_params = 2 × n_embd                    # LayerNorm (weight + bias)

layer_params = attn_params + ffn_params + 2 × ln_params  # Per layer
total_layers_params = n_layer × layer_params

total ≈ vocab_params + pos_params + total_layers_params
```

For GPT-2 small (124M):
```
= (50257 × 768) + (1024 × 768) + 12 × (4 × 768² + 2 × 768 × 3072 + 4 × 768)
≈ 38.6M + 0.8M + 85.0M
≈ 124.4M parameters
```

## Next Steps

Once you pass all tests, move on to Problem 10: Full GPT-2 Model Assembly, where we'll use this configuration to build the complete model!
