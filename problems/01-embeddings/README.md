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

### Position Embeddings

Unlike RNNs, transformers don't have an inherent sense of position or order. Position embeddings solve this by learning a unique vector for each position in the sequence.

Position 0 → [0.2, 0.1, -0.1, ...]
Position 1 → [0.3, -0.2, 0.4, ...]
...and so on

### Combining Embeddings

GPT-2 uses a simple approach: add token embeddings and position embeddings element-wise. This gives the model information about both what the token is AND where it appears in the sequence.

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
- **Dropout**: Regularization technique that randomly zeros some elements during training
- **Device placement**: Ensuring tensors are on the same device (CPU/GPU)

## Next Steps

Once you pass all tests, move on to **Problem 2: Attention Basics**!

The embeddings you've implemented here will be the foundation for the rest of the GPT-2 model. Every input sequence starts by going through this embedding layer.
