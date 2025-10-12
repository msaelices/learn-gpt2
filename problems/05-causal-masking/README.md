# Problem 5: Causal Masking

## Learning Objectives
- Understand autoregressive language modeling
- Learn why we prevent attending to future tokens
- Implement triangular causal mask
- Handle masking in attention computation
- Understand the difference between causal and bidirectional attention

## Background

In Problems 2-4, we implemented attention mechanisms that allow tokens to attend to **all** other tokens in the sequence, including future ones. This is called **bidirectional attention** and works great for tasks like BERT where you have the full sentence available.

However, GPT-2 is an **autoregressive language model** - it generates text one token at a time, predicting the next token based only on previous tokens. During training and generation, a token should **never** see future tokens, as they wouldn't be available during actual text generation.

### Why Causal Masking?

**The Problem**: Without masking, the model could "cheat" during training by looking at future tokens to predict the current token.

**The Solution**: Apply a **causal mask** (also called an autoregressive mask) that prevents positions from attending to subsequent positions.

### The Causal Mask

A causal mask is a **lower-triangular matrix** filled with 1s below the diagonal and 0s above:

```
Position:  0  1  2  3  4
        0 [1  0  0  0  0]   Position 0 can only see itself
        1 [1  1  0  0  0]   Position 1 can see 0 and 1
        2 [1  1  1  0  0]   Position 2 can see 0, 1, and 2
        3 [1  1  1  1  0]   Position 3 can see 0, 1, 2, and 3
        4 [1  1  1  1  1]   Position 4 can see all previous positions
```

### How to Apply the Mask

Before applying softmax to attention scores, we set masked positions to `-inf`:

```python
# Attention scores: (batch, n_head, seq_len, seq_len)
attn_scores = (Q @ K^T) / sqrt(d_k)

# Apply causal mask (set future positions to -inf)
attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

# After softmax, -inf becomes 0
attn_weights = softmax(attn_scores, dim=-1)
```

When `-inf` goes through softmax, it becomes 0, effectively preventing attention to those positions.

### Architecture

```
Input: (batch, seq_len, n_embd)
   ↓
Multi-Head Attention (from Problem 4)
   ↓
Compute attention scores: (Q @ K^T) / √d_k
   ↓
Apply causal mask (set future positions to -inf)
   ↓
Softmax → masked positions become 0
   ↓
Apply attention to values
   ↓
Output: (batch, seq_len, n_embd)
```

## Your Task

Extend the `MultiHeadAttention` class from Problem 4 to include causal masking:

1. Register a causal mask buffer in `__init__`
2. Apply the mask to attention scores before softmax in `forward`
3. Ensure the mask works with variable sequence lengths

## Hints

💡 **Getting Started**
- Start with your `MultiHeadAttention` from Problem 4
- Use `torch.tril()` to create a lower-triangular mask
- Register it as a buffer with `self.register_buffer()` so it moves with the model to GPU/CPU
- Create the mask with shape `(1, 1, n_positions, n_positions)` to broadcast correctly

💡 **Implementation Tips**
- Create mask: `torch.tril(torch.ones(n_positions, n_positions))`
- Shape it for broadcasting: `.view(1, 1, n_positions, n_positions)`
- Apply before softmax: `attn_scores.masked_fill(mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))`
- Slice the mask to match current sequence length: `mask[:, :, :seq_len, :seq_len]`
- The mask is created once and reused for all sequences

💡 **Common Pitfalls**
- Don't forget to slice the mask for shorter sequences than `n_positions`
- Apply the mask **before** softmax, not after
- Use `float('-inf')`, not a large negative number like `-1e9`
- Make sure mask is on the same device as attention scores
- The mask shape must broadcast with attention scores: `(1, 1, seq_len, seq_len)` broadcasts to `(batch, n_head, seq_len, seq_len)`

💡 **Testing Tips**
- Verify that `attn_weights[i, j]` is 0 when `j > i` (future positions)
- Check that `attn_weights[i, :i+1]` sums to 1.0 (only attends to valid positions)
- Test with different sequence lengths
- Compare causal vs non-causal attention patterns visually

## Testing Your Solution

```bash
cd problems/05-causal-masking
python -m pytest test_causal_masking.py -v
```

## Resources

📚 **PyTorch Documentation**
- [torch.tril](https://pytorch.org/docs/stable/generated/torch.tril.html) - Create lower-triangular matrix
- [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html) - Create tensor of ones
- [Module.register_buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) - Register non-trainable tensor
- [Tensor.masked_fill](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html) - Fill tensor where mask is True
- [torch.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) - Softmax function

📄 **Papers & Articles**
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.2.3 discusses masking
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html) - Detailed walkthrough with causal masking

💻 **Additional Resources**
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Visual explanation of GPT-2
- [Autoregressive Models](https://deepgenerativemodels.github.io/notes/autoregressive/) - Theory of autoregressive modeling

## Key Concepts

**Autoregressive Generation**:
```
Given: "The cat sat on the"
Predict: "mat"

Position 0: "The"     → predicts "cat"    (can only see "The")
Position 1: "cat"     → predicts "sat"    (can see "The cat")
Position 2: "sat"     → predicts "on"     (can see "The cat sat")
Position 3: "on"      → predicts "the"    (can see "The cat sat on")
Position 4: "the"     → predicts "mat"    (can see all previous)
```

**Causal vs Bidirectional**:
- **Causal (GPT-2)**: Token i can only attend to positions ≤ i
- **Bidirectional (BERT)**: Token i can attend to all positions

## Formulas

**Masked Attention**:
```
mask[i,j] = 1 if j ≤ i else 0

attn_scores = (Q @ K^T) / √d_k

attn_scores_masked = where(mask == 1, attn_scores, -inf)

attn_weights = softmax(attn_scores_masked, dim=-1)

output = attn_weights @ V
```

## Next Steps

Once you pass all tests, move on to Problem 6: Feedforward Network (MLP), where we'll implement the position-wise feedforward layer!
