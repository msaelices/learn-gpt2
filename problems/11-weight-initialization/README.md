# Problem 11: Weight Initialization

## Learning Objectives
- Understand the importance of proper weight initialization
- Learn standard initialization schemes for neural networks
- Implement layer-specific initialization strategies
- Ensure training stability from the start
- Match GPT-2's initialization approach

## Background

Weight initialization is critical for training deep neural networks. Poor initialization can lead to:
- **Vanishing gradients**: Signals become too small, learning stops
- **Exploding gradients**: Signals become too large, training diverges
- **Slow convergence**: Model takes much longer to train
- **Poor performance**: Model may not reach optimal solution

### Why Random Initialization Matters

When you create a neural network with PyTorch, weights are initialized randomly by default. However, the default initialization may not be optimal for all architectures. GPT-2 uses a specific initialization scheme that has been carefully tuned for transformer models.

### GPT-2 Initialization Strategy

GPT-2 uses the following initialization scheme:

1. **Linear layers**:
   - Weights: Normal distribution with mean=0, std=0.02
   - Biases: Zeros

2. **Embedding layers**:
   - Weights: Normal distribution with mean=0, std=0.02

3. **LayerNorm layers**:
   - Weights (gain): Ones
   - Biases (bias): Zeros

### The Standard Deviation Choice

Why std=0.02? This is a relatively small value that:
- Prevents activation from being too large initially
- Allows gradients to flow properly
- Has been empirically validated for transformer models
- Balances between too small (vanishing) and too large (exploding)

### Layer-Specific Initialization

Different layer types need different initialization:

```python
if isinstance(module, nn.Linear):
    # Linear transformations need small random weights
    nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

elif isinstance(module, nn.Embedding):
    # Embeddings need small random weights
    nn.init.normal_(module.weight, mean=0.0, std=0.02)

elif isinstance(module, nn.LayerNorm):
    # LayerNorm starts with no scaling (1) and no shift (0)
    nn.init.ones_(module.weight)
    nn.init.zeros_(module.bias)
```

## Your Task

Add a weight initialization method to the `GPT2Model` class that:
1. Implements the `_init_weights` method for different layer types
2. Applies initialization to all modules using `self.apply()`
3. Ensures all weights are properly initialized before training

The `apply()` method is a PyTorch utility that recursively applies a function to all modules in the model, making it perfect for initialization.

## Hints

💡 **Getting Started**
- Add `_init_weights` method that takes a module as parameter
- Use `isinstance()` to check module type
- Call `self.apply(self._init_weights)` at the end of `__init__`

💡 **Implementation Tips**
- Use `torch.nn.init.normal_(tensor, mean, std)` for normal initialization
- Use `torch.nn.init.zeros_(tensor)` to initialize to zero
- Use `torch.nn.init.ones_(tensor)` to initialize to one
- Check if bias exists before initializing: `if module.bias is not None`
- The underscore in init functions (like `normal_`) means in-place operation

💡 **Common Pitfalls**
- Don't forget to check if bias exists before initializing it
- Make sure to call `self.apply()` at the END of `__init__`, after all modules are created
- Use mean=0.0 and std=0.02 for GPT-2 (not other values)
- LayerNorm weights should be ones, not zeros

💡 **Testing Tips**
- Check weight statistics: mean should be ≈0, std should be ≈0.02
- Verify biases are exactly zero
- Ensure LayerNorm weights are exactly one
- Run forward pass to ensure no NaN or inf values

## Testing Your Solution

```bash
cd problems/11-weight-initialization
uv run pytest test_initialization.py -v
```

## Resources

📚 **PyTorch Documentation**
- [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) - Initialization functions
- [Module.apply](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply) - Apply function to all modules
- [isinstance](https://docs.python.org/3/library/functions.html#isinstance) - Check object type

📄 **Papers & Articles**
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) - Xavier/Glorot initialization
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) - He initialization
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper

💻 **Additional Resources**
- [Weight Initialization in Neural Networks](https://www.deeplearning.ai/ai-notes/initialization/)
- [Understanding Weight Initialization](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)

## Key Concepts

**Why Not Zero Initialization?**
```python
# BAD: All weights zero
nn.init.zeros_(layer.weight)  # ❌ All neurons compute same thing!

# GOOD: Small random weights
nn.init.normal_(layer.weight, mean=0.0, std=0.02)  # ✓ Breaks symmetry
```

If all weights start at zero (or any same value), all neurons in a layer will:
- Compute the same output
- Receive the same gradients
- Update in the same way
- Never learn different features (symmetry problem)

**The Role of Standard Deviation**

```python
# Too small (std=0.0001)
# - Activations → 0
# - Gradients vanish
# - Learning is very slow

# Just right (std=0.02)
# - Activations are reasonable
# - Gradients flow well
# - Stable learning

# Too large (std=1.0)
# - Activations explode
# - Gradients unstable
# - Training diverges
```

**Module.apply() Pattern**

The `apply()` method recursively visits every module:

```python
def _init_weights(self, module):
    """This function is called for EVERY module."""
    if isinstance(module, nn.Linear):
        # Initialize linear layers
        ...
    elif isinstance(module, nn.Embedding):
        # Initialize embeddings
        ...

# Call it once, applies to all modules
self.apply(self._init_weights)
```

**Initialization Order Matters**

```python
def __init__(self, config):
    super().__init__()

    # 1. Create all modules first
    self.wte = nn.Embedding(...)
    self.wpe = nn.Embedding(...)
    self.h = nn.ModuleList([...])
    self.ln_f = nn.LayerNorm(...)
    self.lm_head = nn.Linear(...)

    # 2. Tie weights (if needed)
    self.lm_head.weight = self.wte.weight

    # 3. Initialize weights LAST
    self.apply(self._init_weights)  # ← Must be last!
```

## Initialization Schemes Comparison

Different papers propose different initialization schemes:

| Scheme | Formula | Best For | Notes |
|--------|---------|----------|-------|
| **Xavier/Glorot** | std = √(2/(fan_in + fan_out)) | Tanh, Sigmoid | Maintains variance |
| **He/Kaiming** | std = √(2/fan_in) | ReLU networks | Accounts for ReLU |
| **GPT-2** | std = 0.02 | Transformers | Empirically tuned |

GPT-2 uses a fixed std=0.02 rather than fan-based initialization. This simpler approach works well for transformers.

## Weight Statistics After Initialization

For a properly initialized model:

```python
model = GPT2Model(config)

# Check linear layer
layer = model.h[0].attn.c_attn
print(f"Weight mean: {layer.weight.mean():.6f}")  # Should be ≈ 0.0
print(f"Weight std: {layer.weight.std():.6f}")    # Should be ≈ 0.02
print(f"Bias mean: {layer.bias.mean():.6f}")      # Should be 0.0

# Check embedding layer
print(f"Embedding mean: {model.wte.weight.mean():.6f}")  # Should be ≈ 0.0
print(f"Embedding std: {model.wte.weight.std():.6f}")    # Should be ≈ 0.02

# Check LayerNorm
ln = model.h[0].ln_1
print(f"LN weight mean: {ln.weight.mean():.6f}")  # Should be 1.0
print(f"LN bias mean: {ln.bias.mean():.6f}")      # Should be 0.0
```

## Example: Before and After Initialization

**Before (Default PyTorch initialization):**
```python
model = GPT2Model(config)
# Linear layers: Some default initialization (varies by layer)
# Embeddings: uniform(-1, 1) / sqrt(embedding_dim)
# LayerNorm: weight=1, bias=0 (actually correct by default)
```

**After (GPT-2 initialization):**
```python
model = GPT2Model(config)
# Linear layers: normal(0, 0.02)
# Embeddings: normal(0, 0.02)
# LayerNorm: weight=1, bias=0
# All layers use consistent initialization scheme
```

## Training Stability

Proper initialization leads to:
- ✓ Stable activations from the start
- ✓ Good gradient flow (not vanishing or exploding)
- ✓ Faster convergence
- ✓ Better final performance
- ✓ Reproducible results

## Next Steps

Once you pass all tests, move on to Problem 12: Loading Pretrained Weights, where we'll load the actual pretrained GPT-2 weights from HuggingFace!
