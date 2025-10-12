# Problem 8: Complete Transformer Block

## Learning Objectives
- Combine all previous components into a complete transformer block
- Understand information flow through attention and feedforward layers
- Implement pre-norm architecture with residual connections
- See how all pieces work together in the full transformer

## Background

The transformer block is the fundamental building unit of GPT-2. It combines everything we've learned:
- Layer normalization (Problem 7)
- Residual connections (Problem 7)
- Causal multi-head attention (Problems 4 & 5)
- Position-wise feedforward network (Problem 6)

Each transformer block performs two main operations in sequence:
1. **Attention sublayer**: Allows tokens to gather information from other tokens
2. **Feedforward sublayer**: Processes each token independently with a two-layer MLP

Both sublayers use the pre-norm architecture with residual connections for stable training.

### Pre-Norm Architecture

GPT-2 uses **pre-norm** (normalize before sublayer):
```
x = x + attention(layer_norm(x))
x = x + feedforward(layer_norm(x))
```

This differs from the original Transformer's **post-norm** (normalize after sublayer):
```
x = layer_norm(x + attention(x))
x = layer_norm(x + feedforward(x))
```

Pre-norm provides more stable training and is easier to train without warmup.

### Information Flow

```
Input x
  ↓
LayerNorm (ln_1)
  ↓
Multi-Head Attention (with causal mask)
  ↓
Residual Add (+)  ←─────────┐
  ↓                         │
LayerNorm (ln_2)           │
  ↓                         │
Feedforward Network        │
  ↓                         │
Residual Add (+)  ←─────────┘
  ↓
Output x'
```

## Your Task

Implement the `TransformerBlock` class that combines:
- Two layer normalization modules
- Causal multi-head attention
- Position-wise feedforward network
- Two residual connections

The block should implement pre-norm architecture where:
1. First layer norm → attention → residual add
2. Second layer norm → feedforward → residual add

## Hints

💡 **Getting Started**
- You'll use the `MultiHeadAttention` and `FeedForward` classes from previous problems
- Create two separate `nn.LayerNorm` modules: one before attention, one before feedforward
- Each sublayer (attention and feedforward) should be wrapped with its own residual connection

💡 **Implementation Tips**
- Initialize two layer norms: `ln_1` (before attention) and `ln_2` (before feedforward)
- Use `MultiHeadAttention` with causal masking built-in
- Use `FeedForward` for the MLP layer
- Apply pre-norm pattern: `x = x + sublayer(layer_norm(x))`
- The attention layer needs causal masking, but the feedforward doesn't

💡 **Common Pitfalls**
- Don't forget the residual connections (the `+` operation)
- Pre-norm means layer norm comes BEFORE the sublayer, not after
- Make sure to use two separate layer norm instances (don't reuse the same one)
- The attention sublayer uses the causal mask; feedforward doesn't need masking
- Input and output shapes must match: `(batch_size, seq_len, n_embd)`

💡 **Testing Tips**
- Verify that input and output shapes are identical
- Check that gradients flow through both residual paths
- Test with different sequence lengths
- Ensure causal masking is working (future tokens don't affect past)
- Verify layer norms are normalizing correctly

## Testing Your Solution

```bash
cd problems/08-transformer-block
python -m pytest test_transformer_block.py -v
```

## Resources

📚 **PyTorch Documentation**
- [nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) - Layer normalization
- [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) - Base class for neural network modules

📄 **Papers & Articles**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer (post-norm architecture)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-norm vs post-norm analysis
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

💻 **Additional Resources**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide to transformer architecture
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html) - Detailed GPT-2 walkthrough

## Key Concepts

**Why Two Separate Layer Norms?**
- Each sublayer (attention and feedforward) has its own layer norm
- This allows independent normalization before each transformation
- Improves training stability and model capacity

**Pre-Norm Benefits**:
- More stable training (less sensitive to hyperparameters)
- Easier to train deep networks without warmup
- Gradients flow more smoothly
- Widely adopted in modern transformers (GPT-2, GPT-3, etc.)

**Residual Connections**:
- Provide direct gradient paths
- Enable training of very deep networks
- Help preserve information from lower layers
- Mathematically: ensure gradient has a "+1" component

## Architecture Diagram

**Single Transformer Block**:
```
     Input: (batch, seq_len, n_embd)
            ↓
      ┌─────────────────┐
      │  LayerNorm 1    │
      └─────────────────┘
            ↓
      ┌─────────────────┐
      │  Multi-Head     │
      │  Attention      │  (with causal mask)
      │  (n_head heads) │
      └─────────────────┘
            ↓
         Residual (+) ←───────┐
            ↓                 │
      ┌─────────────────┐    │
      │  LayerNorm 2    │    │
      └─────────────────┘    │
            ↓                 │
      ┌─────────────────┐    │
      │  Feedforward    │    │
      │  (4x expansion) │    │
      └─────────────────┘    │
            ↓                 │
         Residual (+) ←───────┘
            ↓
     Output: (batch, seq_len, n_embd)
```

**Stacked Blocks (Full GPT-2)**:
```
Input Embeddings
      ↓
[ Block 0 ]  ← 12 blocks for GPT-2 small
[ Block 1 ]  ← 24 blocks for GPT-2 medium
[ Block 2 ]  ← 36 blocks for GPT-2 large
    ...
[ Block 11]  ← 48 blocks for GPT-2 xl
      ↓
 Final LayerNorm
      ↓
 LM Head (projection to vocab)
      ↓
   Logits
```

## Next Steps

Once you pass all tests, move on to Problem 9: GPT-2 Configuration, where we'll organize all hyperparameters and create presets for different model sizes (small, medium, large, xl)!
