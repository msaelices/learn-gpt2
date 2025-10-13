# Problem 1: Token & Position Embeddings

⭐ **Difficulty**: Easy
⏱️ **Time estimate**: 30-45 minutes

## Learning Objectives

- Understand what embeddings are and why we need them in language models
- Learn about token embeddings (vocabulary indices → dense vectors)
- Learn about position embeddings (sequence positions → dense vectors)
- Understand how to combine embeddings and apply dropout

## Background

In natural language processing, we need to convert discrete tokens (words or subwords) into continuous vector representations that neural networks can process. This is where embeddings come in.

### Token Embeddings

Token embeddings map each vocabulary item (token) to a learned vector representation. For example, if our vocabulary size is 50,257 and embedding dimension is 768, then each token gets its own 768-dimensional vector.

Think of it as a lookup table: token ID 42 → [0.1, -0.3, 0.5, ..., 0.2] (768 numbers)

**Why Dense Vectors?**
Neural networks can't process discrete symbols (words/tokens) directly. Embeddings convert these discrete symbols into continuous vectors that:
- Capture semantic relationships (similar words have similar vectors)
- Can be learned through backpropagation during training
- Reduce dimensionality compared to one-hot encoding (768 dimensions vs 50,257)
- Enable mathematical operations like addition and similarity computation

**Learned vs Fixed**
Unlike older methods (e.g., Word2Vec), GPT-2's token embeddings are learned end-to-end as part of model training. They start random and gradually learn to represent tokens in ways useful for language modeling.

### Position Embeddings

Unlike RNNs, transformers don't have an inherent sense of position or order. Position embeddings solve this by learning a unique vector for each position in the sequence.

Position 0 → [0.2, 0.1, -0.1, ...]
Position 1 → [0.3, -0.2, 0.4, ...]
...and so on

**Why Are Position Embeddings Necessary?**
Without position information, the transformer would treat "dog bites man" identically to "man bites dog" - the attention mechanism is inherently permutation-invariant. Position embeddings inject crucial word order information.

**GPT-2's Approach**
GPT-2 uses *learned* position embeddings rather than sinusoidal (fixed) encodings used in the original Transformer:
- Each position (0 to 1023) gets its own learnable vector
- These are trained alongside the model, allowing them to adapt to the specific task
- They're completely separate from token embeddings - position 0 has the same embedding regardless of which token appears there
- Maximum sequence length is fixed at training time (1024 for GPT-2)

### Combining Embeddings

GPT-2 uses a simple approach: add token embeddings and position embeddings element-wise. This gives the model information about both what the token is AND where it appears in the sequence.

**Why Addition?**
You might wonder why we add rather than concatenate. Addition has several advantages:
- **Preserves dimensionality**: Output has the same dimension (768) as input, keeping the model architecture clean
- **Efficient**: No additional parameters needed for projection after concatenation
- **Effective**: Despite being simple, addition allows the model to learn how position modifies meaning
- **Mathematically flexible**: The subsequent transformer layers can learn to separate or combine these signals as needed

For example:
```
Token "cat" embedding:     [0.5, -0.3, 0.8, ...]
Position 2 embedding:      [0.1,  0.2, -0.1, ...]
Combined embedding:        [0.6, -0.1, 0.7, ...]  # Element-wise sum
```

The resulting vector represents "cat at position 2" in a single unified representation.

### Dropout in Embeddings

After combining the embeddings, we apply dropout - a crucial regularization technique. Here's why it matters:

**What is Dropout?**
Dropout randomly sets a fraction of embedding values to zero during training (typically 10% in GPT-2). For example, if an embedding is `[0.5, 0.3, -0.2, 0.8]`, dropout might convert it to `[0.5, 0.0, -0.2, 0.8]`.

**Why Apply Dropout to Embeddings?**
1. **Prevents Overfitting**: Without dropout, the model might memorize specific embedding patterns from the training data
2. **Encourages Robustness**: By randomly dropping dimensions, the model can't rely on any single feature and must learn more distributed representations
3. **Reduces Co-adaptation**: Forces the model to learn redundant representations across multiple dimensions
4. **Better Generalization**: Models with embedding dropout typically perform better on unseen data

**Training vs. Evaluation**
- During training: Dropout is active, randomly zeroing elements with probability `embd_pdrop` (0.1 for GPT-2)
- During evaluation: Dropout is automatically disabled, passing embeddings through unchanged
- PyTorch handles this automatically via `model.train()` and `model.eval()`

**Important Note**: The dropout layer scales the remaining values by `1/(1-p)` during training to maintain the expected sum. For example, with 10% dropout, remaining values are scaled by ~1.11x. This ensures consistent activation magnitudes between training and inference.

## Your Task

Implement an embedding layer that:

1. Takes input token IDs (integers)
2. Looks up token embeddings
3. Creates position IDs for the sequence
4. Looks up position embeddings
5. Combines both embeddings
6. Applies dropout for regularization

Complete the `Embeddings` class in `problem.py`.

## Hints

💡 **Getting Started**
- Use `nn.Embedding(num_embeddings, embedding_dim)` for both token and position embeddings
- Token embeddings map vocabulary indices to vectors
- Position embeddings map positions (0, 1, 2, ...) to vectors

💡 **Implementation Tips**
- Create position indices using `torch.arange(0, seq_len)` on the same device as input_ids
- Get batch size and sequence length from input_ids: `batch_size, seq_len = input_ids.size()`
- Expand position indices to match batch size: `position_ids.unsqueeze(0).expand(batch_size, seq_len)`
- Combine embeddings with simple addition: `token_emb + position_emb`
- Apply dropout after combining embeddings

💡 **Common Pitfalls**
- Don't forget to put position indices on the correct device (use `input_ids.device`)
- Position embeddings should have capacity for `n_positions` (e.g., 1024), not just current `seq_len`
- Make sure batch dimensions align when combining embeddings
- Remember that embedding layers expect integer inputs (token IDs), not floats

💡 **Testing Tips**
- Print shapes at each step to verify dimensions: `print(f"Token emb shape: {token_emb.shape}")`
- Check that changing input tokens changes the output
- Verify dropout is only active in training mode: use `model.train()` vs `model.eval()`
- Test with different batch sizes and sequence lengths

## Testing Your Solution

```bash
cd problems/01-embeddings
uv run pytest test_embeddings.py -v
```

Or run tests with more detail:
```bash
uv run pytest test_embeddings.py -v -s
```

## Expected Output Shape

Input: `(batch_size, seq_len)` - tensor of token IDs
Output: `(batch_size, seq_len, n_embd)` - tensor of combined embeddings

Example:
- Input: `torch.randint(0, 50257, (2, 10))` → shape (2, 10)
- Output: shape (2, 10, 768)

## Resources

📚 **PyTorch Documentation**
- [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) - Embedding layer documentation
- [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html) - Create position indices
- [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - Dropout layer
- [Tensor.unsqueeze](https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html) - Add dimensions to tensors
- [Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) - Expand tensor dimensions
- [Tensor.size](https://pytorch.org/docs/stable/generated/torch.Tensor.size.html) - Get tensor dimensions

📄 **Papers & Articles**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.4 (Embeddings and Softmax)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Original GPT-2 paper
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Understanding embeddings visually

💻 **Additional Resources**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation of embeddings in transformers
- [Understanding Word Embeddings](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)

## Concepts to Understand

- **Vocabulary size**: Number of unique tokens (e.g., 50,257 for GPT-2)
- **Embedding dimension**: Size of the dense vector (e.g., 768 for GPT-2 small)
- **Max positions**: Maximum sequence length supported (e.g., 1024 for GPT-2)
- **Dropout**: Regularization technique that randomly zeros elements during training to prevent overfitting. Applied to embeddings with probability 0.1 (10%) in GPT-2. Includes automatic scaling to maintain expected values.
- **Device placement**: Ensuring tensors are on the same device (CPU/GPU)

## Next Steps

Once you pass all tests, move on to **Problem 2: Attention Basics**!

The embeddings you've implemented here will be the foundation for the rest of the GPT-2 model. Every input sequence starts by going through this embedding layer.
