# Problem 2: Attention Basics

⭐⭐ **Difficulty**: Medium
⏱️ **Time estimate**: 45-60 minutes

## Learning Objectives

- Understand the Query-Key-Value (QKV) attention mechanism
- Learn how attention allows tokens to "look at" other tokens
- Implement basic single-head attention (without scaling)
- Understand attention weights and how they're computed

## Background

Attention is the core mechanism that makes transformers powerful. It allows each position in a sequence to attend to (focus on) all positions, computing a weighted combination of their values.

### The Attention Mechanism

At its heart, attention asks three questions for each token:
1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What do I offer?"
3. **Value (V)**: "What information do I contain?"

Why three different vectors? Because they serve different purposes:
- **Query**: Represents what the current token wants to know
- **Key**: Represents what each token can provide
- **Value**: The actual information content of each token

Why do we need a Value vector if we have the input information? Because the Value vector can be transformed differently from the input, allowing the model to learn what information is most relevant to pass along.

So the X would be the raw input embeddings, and Q, K, V are learned projections of X.

### How It Works

1. Each input token is projected into three vectors: Q, K, and V
   - What projections? Linear layers (learned weight matrices)
2. Compute attention scores: How much should each token attend to every other token?
   - Score = Q @ K^T (dot product between queries and keys)
3. Apply softmax to get attention weights (probabilities that sum to 1)
4. Compute weighted sum of values using attention weights

This creates a context-aware representation where each token incorporates information from other tokens it "attends to."

### Example

Imagine the sentence: "The cat sat on the mat"

For the word "sat":
- Its **Query** might ask: "What is doing the sitting?"
- The word "cat"'s **Key** would match this query
- The attention weight between "sat" and "cat" would be high
- "cat"'s **Value** (its semantic information) gets incorporated into "sat"'s representation

## Your Task

Implement a simple single-head attention mechanism that:

1. Projects input to Q, K, V using linear layers
2. Computes attention scores (Q @ K^T)
3. Applies softmax to get attention weights
4. Computes output as weighted sum of values

Note: This is a simplified version without scaling (we'll add that in Problem 3).

Complete the `SimpleAttention` class in `problem.py`.

## Hints

💡 **Getting Started**
- Use three separate `nn.Linear` layers to project input to Q, K, V
- All three projections should have the same dimensionality as the input (n_embd → n_embd)

💡 **Implementation Tips**
- Compute Q, K, V by passing input through the three linear layers
- Attention scores: `scores = q @ k.transpose(-2, -1)`  (batch matrix multiply)
- Use `torch.softmax(scores, dim=-1)` to normalize scores into weights
- Final output: `output = attention_weights @ v`
- The `@` operator performs batch matrix multiplication

💡 **Common Pitfalls**
- Don't forget to transpose K when computing scores: use `k.transpose(-2, -1)`
- Softmax should be applied on the last dimension (dim=-1) to normalize across keys
- Make sure all matrix dimensions align: (batch, seq_len, n_embd)
- The output should have the same shape as the input

💡 **Testing Tips**
- Verify that attention weights sum to 1 along the last dimension
- Check that output shape equals input shape
- Test with different sequence lengths to ensure it works generally
- Print intermediate shapes to debug dimension mismatches

## Testing Your Solution

```bash
cd problems/02-attention-basics
uv run pytest test_attention.py -v
```

Or run tests with more detail:
```bash
uv run pytest test_attention.py -v -s
```

## Expected Shapes

Input: `(batch_size, seq_len, n_embd)`

Intermediate shapes:
- Q: `(batch_size, seq_len, n_embd)`
- K: `(batch_size, seq_len, n_embd)`
- V: `(batch_size, seq_len, n_embd)`
- Attention scores: `(batch_size, seq_len, seq_len)`
- Attention weights: `(batch_size, seq_len, seq_len)` (after softmax, sums to 1)

Output: `(batch_size, seq_len, n_embd)`

## Resources

📚 **PyTorch Documentation**
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Linear transformation layer
- [torch.matmul / @](https://pytorch.org/docs/stable/generated/torch.matmul.html) - Matrix multiplication
- [Tensor.transpose](https://pytorch.org/docs/stable/generated/torch.Tensor.transpose.html) - Transpose dimensions
- [torch.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) - Softmax function

📄 **Papers & Articles**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.2.1 (Scaled Dot-Product Attention)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation of attention
- [Visualizing Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) - How attention works step by step

💻 **Additional Resources**
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Comprehensive overview
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Line-by-line implementation guide

## Concepts to Understand

- **Self-attention**: Each position attends to all positions in the same sequence
- **Query, Key, Value**: Three different projections of the same input
- **Attention scores**: Raw dot products measuring similarity
- **Attention weights**: Normalized scores (via softmax) that sum to 1
- **Weighted aggregation**: Combining values based on attention weights
- **Permutation invariance** (almost): Without positional info, order doesn't matter

## Visualization Idea

The attention weights form a matrix where:
- Rows represent "from" positions (queries)
- Columns represent "to" positions (keys)
- Each cell shows how much position i attends to position j

This can be visualized as a heatmap showing which tokens focus on which other tokens!

## Next Steps

Once you pass all tests, move on to **Problem 3: Scaled Dot-Product Attention**!

There we'll add the scaling factor (√d_k) which is crucial for numerical stability and proper gradient flow.
