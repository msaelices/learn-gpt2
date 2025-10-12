# Problem 3: Scaled Dot-Product Attention

⭐⭐ **Difficulty**: Medium
⏱️ **Time estimate**: 30-45 minutes

## Learning Objectives

- Understand why scaling is crucial for attention mechanisms
- Learn about numerical stability in deep learning
- Implement scaled dot-product attention with the √d_k factor
- Understand gradient flow and vanishing gradients

## Background

In Problem 2, we implemented basic attention without scaling. However, when the embedding dimension (d_k) is large, the dot products grow large in magnitude, causing the softmax function to have extremely small gradients. This is the **vanishing gradient problem**.

### The Scaling Factor

The **Scaled Dot-Product Attention** divides the attention scores by √d_k before applying softmax:

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

### Why √d_k?

When Q and K have dimension d_k with unit variance:
- The dot product Q @ K^T has variance proportional to d_k
- Dividing by √d_k normalizes the variance back to ~1
- This keeps the softmax input in a reasonable range
- Gradients flow better during backpropagation

### Example

Without scaling (d_k = 512):
- Dot products might range from -50 to +50
- softmax(50) ≈ 1, softmax(-50) ≈ 0 (saturated)
- Gradients ≈ 0 (can't learn)

With scaling (divide by √512 ≈ 22.6):
- Dot products range from -2.2 to +2.2
- softmax works in its linear region
- Gradients flow properly

## Your Task

Modify the SimpleAttention from Problem 2 to add scaling:

1. Calculate the scaling factor: `sqrt(d_k)` where d_k is the dimension per head
2. Scale attention scores before softmax: `scores / sqrt(d_k)`
3. Everything else remains the same

Complete the `ScaledAttention` class in `problem.py`.

## Hints

💡 **Getting Started**
- Copy your SimpleAttention solution from Problem 2
- The only change is adding scaling before softmax
- d_k is the last dimension of the query/key tensors

💡 **Implementation Tips**
- Compute scaling factor: `scale = q.size(-1) ** -0.5` (equivalent to 1/√d_k)
- Scale scores: `attn_scores = (q @ k.transpose(-2, -1)) * scale`
- Alternative: `attn_scores = attn_scores / math.sqrt(d_k)`
- Use `import math` for `math.sqrt()` if needed
- The rest of the forward pass is identical to Problem 2

💡 **Common Pitfalls**
- Don't forget to scale BEFORE softmax (not after!)
- Use `** -0.5` for efficiency instead of `1 / math.sqrt()`
- Make sure to get d_k from the tensor dimension, not hardcode it
- The scaling factor is constant (doesn't change during training)

💡 **Testing Tips**
- Compare outputs with and without scaling
- Verify that larger d_k values still produce reasonable attention weights
- Check that gradients are not vanishing (magnitude > 1e-8)
- Test with different embedding dimensions (64, 256, 512, 1024)

## Testing Your Solution

```bash
cd problems/03-scaled-attention
uv run pytest test_scaled_attention.py -v
```

Or run with more detail:
```bash
uv run pytest test_scaled_attention.py -v -s
```

## Expected Shapes

All shapes are identical to Problem 2:

Input: `(batch_size, seq_len, n_embd)`

Intermediate shapes:
- Q: `(batch_size, seq_len, n_embd)`
- K: `(batch_size, seq_len, n_embd)`
- V: `(batch_size, seq_len, n_embd)`
- Attention scores (before scaling): `(batch_size, seq_len, seq_len)`
- Scaling factor: scalar (1 / √n_embd)
- Attention scores (after scaling): `(batch_size, seq_len, seq_len)`
- Attention weights: `(batch_size, seq_len, seq_len)`

Output: `(batch_size, seq_len, n_embd)`

## Resources

📚 **PyTorch Documentation**
- [torch.Tensor.size](https://pytorch.org/docs/stable/generated/torch.Tensor.size.html) - Get tensor dimensions
- [Tensor operations](https://pytorch.org/docs/stable/torch.html#math-operations) - Mathematical operations
- [math.sqrt](https://docs.python.org/3/library/math.html#math.sqrt) - Square root function

📄 **Papers & Articles**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.2.1 (explains scaling factor)
- [Understanding the Scaled Dot-Product Attention](https://www.baeldung.com/cs/attention-scaled-dot-product) - Detailed explanation
- [Why Scale by √d_k?](https://stats.stackexchange.com/questions/421935/what-is-the-positional-encoding-in-the-transformer-model) - Mathematical justification

💻 **Additional Resources**
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Implementation walkthrough
- [Softmax Saturation](https://paperswithcode.com/method/scaled) - Explains the saturation problem

## Concepts to Understand

- **Scaling factor**: 1/√d_k normalizes variance of dot products
- **Numerical stability**: Keeping values in a reasonable range for computation
- **Gradient flow**: How gradients propagate during backpropagation
- **Vanishing gradients**: When gradients become too small to learn effectively
- **Softmax saturation**: When softmax outputs are too close to 0 or 1
- **Variance normalization**: Keeping statistical properties consistent across layers

## Visualization Idea

Compare attention weights with and without scaling for different d_k values:
- Small d_k (64): Scaling makes little difference
- Medium d_k (256): Scaling helps noticeably
- Large d_k (1024): Scaling is essential (without it, weights are too sharp)

You can visualize this as heatmaps showing attention distributions!

## Next Steps

Once you pass all tests, move on to **Problem 4: Multi-Head Attention**!

There we'll learn how to run multiple attention heads in parallel, allowing the model to attend to different aspects of the input simultaneously.
