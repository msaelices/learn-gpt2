# Problem 6: Feedforward Network (MLP)

## Learning Objectives
- Understand the role of feedforward networks in transformers
- Learn about the 4x expansion factor
- Implement the GELU activation function
- Apply dropout for regularization

## Background

After the attention mechanism processes the input, each transformer block includes a **position-wise feedforward network** (also called MLP - Multi-Layer Perceptron). This is applied independently to each position in the sequence.

### Why Do We Need a Feedforward Network?

While attention allows positions to communicate and gather information, the feedforward network provides:
1. **Non-linear transformation**: Additional capacity to learn complex patterns
2. **Position-wise processing**: Each position is processed independently
3. **Expansion and compression**: Temporarily expands the representation to 4x the size, then projects back

### Architecture

The feedforward network is a simple two-layer network:

```
Input: (batch_size, seq_len, n_embd)
   ↓
Linear expansion: n_embd → n_inner (4 * n_embd)
   ↓
GELU activation (non-linearity)
   ↓
Linear projection: n_inner → n_embd
   ↓
Dropout
   ↓
Output: (batch_size, seq_len, n_embd)
```

For GPT-2 small (n_embd=768):
- Expansion: 768 → 3072 (4x)
- Projection: 3072 → 768

### GELU Activation

GPT-2 uses **GELU** (Gaussian Error Linear Unit) instead of ReLU. GELU is a smooth, non-monotonic activation function that approximates:

```python
GELU(x) = x * Φ(x)
```

where Φ(x) is the cumulative distribution function of the standard normal distribution.

GPT-2 uses an approximation:
```python
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

GELU advantages:
- Smoother than ReLU (no hard cutoff at 0)
- Non-monotonic (can output negative values)
- Better gradient flow
- Empirically performs better for language tasks

## Your Task

Implement two classes:

1. **NewGELU**: The GELU activation function using the approximation formula
2. **FeedForward**: The two-layer feedforward network with GELU

## Hints

💡 **Getting Started**
- NewGELU is just a forward method implementing the formula
- Use `torch.tanh()` and `torch.pow()` for the approximation
- For FeedForward: two `nn.Linear` layers + GELU + dropout

💡 **Implementation Tips**
- NewGELU formula: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
- Use `torch.sqrt(torch.tensor(2.0 / torch.pi))` for the constant
- FeedForward structure: `Linear(n_embd, n_inner) → GELU → Linear(n_inner, n_embd) → Dropout`
- n_inner is typically 4 * n_embd
- Apply dropout AFTER the second linear layer

💡 **Common Pitfalls**
- Don't forget the 0.5 multiplier in GELU
- The cubic term is `0.044715 * x³`, not `0.044715 * x`
- Dropout goes at the end, not between the two linear layers
- Make sure shapes are preserved: input shape = output shape

💡 **Testing Tips**
- Test GELU with known values (e.g., GELU(0) ≈ 0, GELU(1) ≈ 0.841)
- Verify output shape matches input shape
- Check that dropout only activates in training mode
- Compare with PyTorch's built-in `nn.GELU()` (should be close)

## Testing Your Solution

```bash
cd problems/06-feedforward-network
python -m pytest test_feedforward.py -v
```

## Resources

📚 **PyTorch Documentation**
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Linear transformation layer
- [torch.tanh](https://pytorch.org/docs/stable/generated/torch.tanh.html) - Hyperbolic tangent
- [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html) - Power function
- [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - Dropout layer
- [nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) - PyTorch's GELU (for comparison)

📄 **Papers & Articles**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.3 (Position-wise Feed-Forward Networks)
- [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415) - Original GELU paper
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

💻 **Additional Resources**
- [GPT-2 Source Code](https://github.com/openai/gpt-2/blob/master/src/model.py#L25-L26) - Original GELU implementation
- [Understanding GELU Activation](https://paperswithcode.com/method/gelu) - Visual explanation

## Key Concepts

**Position-wise**: The same feedforward network is applied to each position independently. This is different from attention, which mixes information across positions.

**Expansion Factor**: The hidden dimension (n_inner) is typically 4x the embedding dimension. This gives the model more capacity to learn complex transformations.

**Why Two Layers?**:
- First layer: Expands to higher dimension (more capacity)
- Second layer: Projects back to original dimension (maintains compatibility)

## Formulas

**GELU (Approximate)**:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**FeedForward**:
```
FFN(x) = Dropout(Linear₂(GELU(Linear₁(x))))

where:
  Linear₁: n_embd → 4*n_embd
  Linear₂: 4*n_embd → n_embd
```

## Architecture Comparison

**ReLU** (used in original Transformer):
```
ReLU(x) = max(0, x)
```
- Sharp cutoff at 0
- Always non-negative
- Dead neurons problem

**GELU** (used in GPT-2, BERT):
```
GELU(x) ≈ x * sigmoid(1.702 * x)  (simpler approximation)
```
- Smooth curve
- Can output small negative values
- Better performance for NLP

## Next Steps

Once you pass all tests, move on to Problem 7: Layer Normalization & Residuals, where we'll add normalization and skip connections!
