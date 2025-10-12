# Problem 4: Multi-Head Attention

## Learning Objectives
- Understand why multiple attention heads are beneficial
- Learn to split embeddings across multiple attention heads
- Implement parallel attention computation
- Master head concatenation and output projection
- Understand the efficiency of combined Q,K,V projections

## Background

In Problem 3, we implemented scaled dot-product attention with a single attention head. However, GPT-2 and most modern transformers use **multi-head attention**, which allows the model to attend to different aspects of the input simultaneously.

### Why Multiple Heads?

Single-head attention computes one set of attention patterns. Multi-head attention runs multiple attention operations in parallel, each potentially learning different relationships:
- One head might focus on syntactic relationships
- Another might capture semantic meaning
- Others might learn positional or structural patterns

By having multiple heads, the model can capture richer, more nuanced relationships in the data.

### Key Concepts

**Head Splitting**: Instead of using the full embedding dimension for attention, we split it across `n_head` heads:
```
head_dim = n_embd / n_head

For GPT-2 small: n_embd=768, n_head=12 → head_dim=64
```

**Parallel Computation**: Each head operates independently on its slice of the embedding dimension, computing its own attention patterns.

**Concatenation**: After computing attention for all heads, we concatenate their outputs back together and project to the original embedding dimension.

### Architecture

```
Input: (batch_size, seq_len, n_embd)
   ↓
Combined Q,K,V Projection: Linear(n_embd → 3*n_embd)
   ↓
Split into Q, K, V: each (batch_size, seq_len, n_embd)
   ↓
Reshape to heads: (batch_size, n_head, seq_len, head_dim)
   ↓
Scaled Dot-Product Attention per head
   ↓
Concatenate heads: (batch_size, seq_len, n_embd)
   ↓
Output Projection: Linear(n_embd → n_embd)
   ↓
Output: (batch_size, seq_len, n_embd)
```

## Your Task

Implement the `MultiHeadAttention` class that:
1. Creates combined Q,K,V projection (more efficient than separate projections)
2. Splits the embeddings across multiple attention heads
3. Computes scaled dot-product attention for each head in parallel
4. Concatenates the head outputs
5. Projects back to the original embedding dimension

**Note**: This problem focuses on multi-head mechanics without causal masking. We'll add causal masking in Problem 5.

## Hints

💡 **Getting Started**
- Validate that `n_embd` is divisible by `n_head` in `__init__`
- Calculate `head_dim = n_embd // n_head`
- Use a single `nn.Linear(n_embd, 3 * n_embd)` for combined Q,K,V projection (more efficient!)
- Use `nn.Linear(n_embd, n_embd)` for output projection

💡 **Implementation Tips**
- After the combined projection, split the result into Q, K, V using `tensor.split(n_embd, dim=2)`
- Reshape from `(batch, seq_len, n_embd)` to `(batch, seq_len, n_head, head_dim)` using `.view()`
- Transpose to `(batch, n_head, seq_len, head_dim)` using `.transpose(1, 2)` for efficient computation
- After attention, transpose back and reshape to concatenate heads: `.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)`
- Scale by `head_dim ** 0.5`, not `n_embd ** 0.5`!

💡 **Common Pitfalls**
- Don't forget to make the tensor contiguous after transpose: `.transpose().contiguous().view()`
- Make sure to scale by `head_dim`, not `n_embd` (each head works with `head_dim` dimensions)
- The attention computation happens on the last two dimensions: `(seq_len, head_dim)`
- Apply dropout to both attention weights and the final output

💡 **Testing Tips**
- Test with `n_head=1` first - should behave like single-head attention
- Print shapes at each step to verify dimensions
- Verify that output shape matches input shape: `(batch_size, seq_len, n_embd)`
- Check that different heads produce different attention patterns

## Testing Your Solution

```bash
cd problems/04-multi-head-attention
python -m pytest test_multi_head_attention.py -v
```

## Resources

📚 **PyTorch Documentation**
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Linear projection layers
- [Tensor.view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) - Reshape tensors
- [Tensor.transpose](https://pytorch.org/docs/stable/generated/torch.Tensor.transpose.html) - Swap dimensions
- [Tensor.contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) - Make tensor memory contiguous
- [Tensor.split](https://pytorch.org/docs/stable/generated/torch.split.html) - Split tensor into chunks
- [torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html) or `@` operator - Matrix multiplication

📄 **Papers & Articles**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.2.2 (Multi-Head Attention), Equation 2-5
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/#self-attention-in-detail) - Visual explanation of multi-head attention
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

💻 **Additional Resources**
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation guide
- [Multi-Head Attention Explained](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853) - Visual deep dive

## Key Formulas

**Multi-Head Attention**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Scaled Dot-Product Attention** (per head):
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V

where d_k = head_dim (not n_embd!)
```

## Next Steps

Once you pass all tests, move on to Problem 5: Causal Masking, where we'll add the causal mask to prevent attending to future tokens!
