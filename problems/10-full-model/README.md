# Problem 10: Full GPT-2 Model Assembly

## Learning Objectives
- Assemble all previous components into a complete model
- Stack transformer blocks to create depth
- Implement the language modeling head
- Understand weight tying between embeddings and output projection
- Handle end-to-end forward pass

## Background

You've built all the individual components of GPT-2! Now it's time to assemble them into the complete model. The full GPT-2 architecture consists of:

1. **Embedding Layer**: Combines token and position embeddings
2. **Transformer Stack**: N identical transformer blocks
3. **Final Layer Norm**: Normalizes the output of the last block
4. **Language Model Head**: Projects back to vocabulary for predictions

### GPT-2 Architecture Overview

```
Input Token IDs: [15496, 995, ...]  (shape: B, T)
                    ↓
┌───────────────────────────────────────┐
│  Token Embeddings (vocab_size → d)   │  ← wte
│  + Position Embeddings (T → d)       │  ← wpe
└───────────────────────────────────────┘
                    ↓
            Embedding Dropout
                    ↓
┌───────────────────────────────────────┐
│         Transformer Block 1           │
│  • LayerNorm                         │
│  • Causal Multi-Head Attention       │
│  • Residual Connection               │
│  • LayerNorm                         │
│  • Feedforward Network               │
│  • Residual Connection               │
└───────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────┐
│         Transformer Block 2           │
│              ...                      │
└───────────────────────────────────────┘
                    ↓
                   ...
                    ↓
┌───────────────────────────────────────┐
│         Transformer Block N           │
│              ...                      │
└───────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────┐
│         Final Layer Norm              │  ← ln_f
└───────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────┐
│      Language Model Head              │  ← lm_head (tied with wte)
│         (d → vocab_size)              │
└───────────────────────────────────────┘
                    ↓
    Logits: (B, T, vocab_size)
```

### Weight Tying

An important optimization in GPT-2 is **weight tying**: the token embedding weights (`wte.weight`) are shared with the language model head (`lm_head.weight`). This:
- Reduces the total parameter count
- Helps the model learn better representations
- Is a common practice in language models

Implementation:
```python
# After creating both layers:
self.lm_head.weight = self.wte.weight  # Share the weights!
```

### Model Sizes

| Model | Layers (N) | Embedding Dim (d) | Heads | Parameters |
|-------|------------|-------------------|-------|------------|
| Small | 12 | 768 | 12 | 124M |
| Medium | 24 | 1024 | 16 | 355M |
| Large | 36 | 1280 | 20 | 774M |
| XL | 48 | 1600 | 25 | 1.5B |

All models use:
- Vocabulary size: 50,257 tokens
- Context length: 1,024 tokens
- FFN expansion: 4× (n_inner = 4 * n_embd)

## Your Task

Implement the `GPT2Model` class that assembles all components into a complete language model. You'll need to:

1. Create embeddings (token + position)
2. Stack N transformer blocks
3. Add final layer normalization
4. Create language modeling head
5. Implement weight tying
6. Connect all components in forward pass

## Hints

💡 **Getting Started**
- Accept a `GPT2Config` object to get all hyperparameters
- Use `nn.ModuleList` to store the transformer blocks
- Don't forget dropout after embeddings

💡 **Implementation Tips**
- Create token embeddings: `nn.Embedding(config.vocab_size, config.n_embd)`
- Create position embeddings: `nn.Embedding(config.n_positions, config.n_embd)`
- Stack blocks in a loop: `nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])`
- Language model head: `nn.Linear(config.n_embd, config.vocab_size, bias=False)`
- Weight tying after creating both: `self.lm_head.weight = self.wte.weight`

💡 **Forward Pass Steps**
1. Get input shape: `batch_size, seq_len = input_ids.shape`
2. Create position IDs: `torch.arange(0, seq_len, device=input_ids.device)`
3. Get token embeddings: `self.wte(input_ids)`
4. Get position embeddings: `self.wpe(position_ids)`
5. Combine and apply dropout: `self.drop(token_emb + pos_emb)`
6. Pass through all transformer blocks sequentially
7. Apply final layer norm
8. Project to vocabulary: `self.lm_head(x)`

💡 **Common Pitfalls**
- Remember to tie weights AFTER creating both embeddings and lm_head
- Position IDs should be on the same device as input_ids
- Don't apply softmax in the model - that's done during loss calculation
- Make sure lm_head has `bias=False`
- Loop through blocks sequentially, not in parallel

💡 **Testing Tips**
- Output shape should be `(batch_size, seq_len, vocab_size)`
- Test with different batch sizes and sequence lengths
- Verify weight tying: `assert model.lm_head.weight is model.wte.weight`
- Check that gradients flow through the entire model
- Test with config objects from Problem 9

## Testing Your Solution

```bash
cd problems/10-full-model
uv run pytest test_gpt2.py -v
```

## Resources

📚 **PyTorch Documentation**
- [nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) - Container for modules
- [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) - Embedding layer
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Linear transformation
- [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html) - Create position indices

📄 **Papers & Articles**
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) - Weight tying paper
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Visual explanation

💻 **Additional Resources**
- [HuggingFace GPT-2 Model](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model)
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html)
- [OpenAI GPT-2 Code](https://github.com/openai/gpt-2/blob/master/src/model.py)

## Key Concepts

**Weight Tying**:
```python
# Instead of two separate weight matrices:
# - wte.weight: (vocab_size, n_embd) for input embeddings
# - lm_head.weight: (vocab_size, n_embd) for output projection
# We share them:
self.lm_head.weight = self.wte.weight

# Benefits:
# 1. Reduces parameters (saves ~38M params for GPT-2 small)
# 2. Helps model learn better token representations
# 3. Input and output share semantic space
```

**Why Stack Transformer Blocks?**
- Each block refines the representation
- Early layers learn syntax and basic patterns
- Later layers learn semantics and complex relationships
- Stacking creates hierarchical representations
- GPT-2 small uses 12 blocks, XL uses 48

**Language Modeling Head**:
```python
# The lm_head projects from embedding space to vocabulary
# Output: logits for each token in vocabulary
logits = lm_head(hidden_states)  # (B, T, d) → (B, T, vocab_size)

# Each position predicts the probability distribution over next token:
# logits[b, t, :] = scores for position t predicting token at t+1
```

**Sequential vs Parallel Processing**:
- Tokens are processed in parallel (batch dimension)
- But transformer blocks are applied sequentially (depth)
- Each block sees the output of the previous block

## Architecture Diagram

```
Component               Shape Transformation
─────────────────────────────────────────────
Input IDs               (B, T)
                          ↓
Token Embeddings        (B, T, d)  ← nn.Embedding(vocab_size, d)
Position Embeddings     (B, T, d)  ← nn.Embedding(n_positions, d)
                          ↓
Combined + Dropout      (B, T, d)
                          ↓
Block 1                 (B, T, d)  ← TransformerBlock
Block 2                 (B, T, d)  ← TransformerBlock
...                     ...
Block N                 (B, T, d)  ← TransformerBlock
                          ↓
Final LayerNorm         (B, T, d)  ← nn.LayerNorm(d)
                          ↓
LM Head                 (B, T, V)  ← nn.Linear(d, vocab_size)
                          ↓
Logits                  (B, T, vocab_size)

Where:
  B = batch_size
  T = sequence_length (seq_len)
  d = n_embd (embedding dimension)
  V = vocab_size
```

## Example Usage

```python
from solution import GPT2Config, GPT2Model
import torch

# Create configuration
config = GPT2Config.gpt2_small()

# Create model
model = GPT2Model(config)

# Create sample input
input_ids = torch.randint(0, config.vocab_size, (2, 10))  # batch=2, seq_len=10

# Forward pass
logits = model(input_ids)

# Output shape
print(f"Input shape: {input_ids.shape}")       # (2, 10)
print(f"Output shape: {logits.shape}")         # (2, 10, 50257)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Parameter Calculation

For GPT-2 small (12 layers, 768 dim, 12 heads):

```
Token embeddings:      50,257 × 768 = 38,597,376
Position embeddings:    1,024 × 768 =    786,432
─────────────────────────────────────────────────
Per transformer block:
  Attention (c_attn):   768 × 2,304 =  1,769,472  (Q, K, V combined)
  Attention (c_proj):   768 × 768   =    589,824
  LayerNorm 1:          768 × 2     =      1,536
  FFN (c_fc):           768 × 3,072 =  2,359,296
  FFN (c_proj):       3,072 × 768   =  2,359,296
  LayerNorm 2:          768 × 2     =      1,536
  ─────────────────────────────────
  Per block total:                  =  7,080,960

12 blocks:            12 × 7,080,960 = 84,971,520
Final LayerNorm:                768 × 2 =      1,536
LM Head:           (tied with embeddings) =          0
─────────────────────────────────────────────────
Total:                            ≈ 124,356,864 params
                                  ≈ 124M parameters
```

## Next Steps

Once you pass all tests, move on to Problem 11: Weight Initialization, where we'll properly initialize all the model weights for training stability!
