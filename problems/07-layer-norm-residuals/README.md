# Problem 7: Layer Normalization & Residuals

## Learning Objectives
- Understand layer normalization and why it's needed
- Learn about residual connections (skip connections)
- Implement pre-norm architecture
- Understand gradient flow through deep networks

## Background

Deep neural networks face two key challenges:
1. **Internal Covariate Shift**: Layer inputs change during training, slowing convergence
2. **Vanishing Gradients**: Gradients diminish through many layers, preventing learning

Two techniques solve these problems:
- **Layer Normalization**: Normalizes activations for stable training
- **Residual Connections**: Provides direct gradient paths through the network

### Layer Normalization

Layer normalization normalizes inputs across the feature dimension, ensuring mean ≈ 0 and variance ≈ 1.

**Formula**:
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

where:
  μ = mean across features
  σ² = variance across features
  γ = learnable scale
  β = learnable shift
  ε = small constant for numerical stability (1e-5)
```

**Benefits**:
- Stabilizes training (prevents explosion/vanishing)
- Reduces training time
- Less sensitive to initialization
- Works well with attention mechanisms

### Residual Connections (Skip Connections)

Residual connections add the input directly to the output of a layer:
```
output = x + SubLayer(x)
```

**Benefits**:
- **Gradient Flow**: Gradients flow directly backwards through the addition
- **Identity Mapping**: Network can learn to skip layers if needed
- **Deeper Networks**: Enables training of very deep networks (100+ layers)

### Pre-Norm vs Post-Norm Architecture

**Post-Norm** (Original Transformer):
```
x = x + SubLayer(LayerNorm(x))
```

**Pre-Norm** (GPT-2, modern transformers):
```
x = x + SubLayer(LayerNorm(x))
```

GPT-2 uses **pre-norm** because it:
- Provides more stable training
- Works better with very deep networks
- Easier to train without learning rate warmup

### In GPT-2 Transformer Block

```
Input
  ↓
LayerNorm (ln_1)
  ↓
Multi-Head Attention
  ↓
Residual Add (+)  ←─────────┐
  ↓                         │
LayerNorm (ln_2)           │
  ↓                         │
Feedforward Network        │
  ↓                         │
Residual Add (+)  ←─────────┘
  ↓
Output
```

## Your Task

For this problem, you'll implement the residual connection pattern that wraps around sublayers (attention or feedforward). The actual application will be shown in Problem 8 when we build the complete transformer block.

Implement a simple helper that demonstrates:
1. Applying layer normalization
2. Passing through a sublayer
3. Adding the residual connection

## Hints

💡 **Getting Started**
- Use `nn.LayerNorm(n_embd, eps=layer_norm_epsilon)` for normalization
- Layer norm normalizes across the last dimension (embedding dimension)
- Residual connection is just addition: `x + sublayer_output`

💡 **Implementation Tips**
- Pre-norm: `x = x + sublayer(ln(x))`
- The `sublayer` is just a function/module that takes x and returns same shape
- No learnable parameters needed beyond LayerNorm's γ and β
- Shape must be preserved: input shape = output shape

💡 **Common Pitfalls**
- Don't forget the addition (`+`) for the residual connection
- Layer norm normalizes the LAST dimension, not batch or sequence
- Pre-norm applies LayerNorm BEFORE the sublayer, not after
- The sublayer must return the same shape as its input

💡 **Testing Tips**
- Verify layer norm normalizes (mean ≈ 0, std ≈ 1)
- Check that residual connection is applied (output ≠ sublayer output alone)
- Test gradient flow through the residual path
- Ensure shapes are preserved throughout

## Testing Your Solution

```bash
cd problems/07-layer-norm-residuals
python -m pytest test_layer_norm.py -v
```

## Resources

📚 **PyTorch Documentation**
- [nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) - Layer normalization
- [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html) - Compute mean
- [torch.std](https://pytorch.org/docs/stable/generated/torch.std.html) - Compute standard deviation

📄 **Papers & Articles**
- [Layer Normalization](https://arxiv.org/abs/1607.06450) - Original LayerNorm paper
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - ResNet paper introducing residual connections
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-norm vs Post-norm analysis
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

💻 **Additional Resources**
- [Understanding Residual Connections](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Section on normalization

## Key Concepts

**Why Layer Norm Instead of Batch Norm?**
- Batch Norm: Normalizes across batch dimension (doesn't work well with variable sequence lengths)
- Layer Norm: Normalizes across feature dimension (works great for sequences)

**Gradient Flow with Residuals**:
```
Without residual: ∂L/∂x = ∂L/∂f(x) * ∂f(x)/∂x
With residual:    ∂L/∂x = ∂L/∂(x+f(x)) * (1 + ∂f(x)/∂x)
                                          ↑
                                    Direct path!
```

The "+1" term ensures gradients can flow directly, even if ∂f(x)/∂x becomes small.

## Formulas

**Layer Normalization**:
```
μ = mean(x, dim=-1)
σ = std(x, dim=-1)
x_normalized = (x - μ) / (σ + ε)
output = γ * x_normalized + β
```

**Pre-Norm Residual Block**:
```
y = LayerNorm(x)
z = SubLayer(y)
output = x + z
```

## Architecture Diagram

```
     Input x
       │
       ├─────────────────┐
       │                 │
   LayerNorm            │
       │                 │
    SubLayer            │
     (Attn/FFN)        │
       │                 │
       └────────(+)──────┘
              │
          Output x'
```

The residual connection allows information to bypass the sublayer entirely if needed.

## Next Steps

Once you pass all tests, move on to Problem 8: Complete Transformer Block, where we'll combine everything (attention, feedforward, layer norm, residuals) into a full transformer block!
