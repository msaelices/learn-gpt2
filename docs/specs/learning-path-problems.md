# Specification: GPT-2 Progressive Learning Path

## Overview

This specification outlines a comprehensive, problem-based learning curriculum for implementing GPT-2 from scratch. The curriculum is designed for developers who want to learn AI and LLMs with PyTorch through hands-on implementation.

## Goals

1. **Progressive Learning**: Break down GPT-2 into digestible, sequential problems
2. **Hands-on Practice**: Each problem includes code to implement, tests to pass, and interactive notebooks
3. **Complete Implementation**: Solving all problems results in a fully functional GPT-2 model
4. **Self-Paced**: Learners can progress at their own pace with clear checkpoints
5. **Validation**: Automated tests ensure correctness at each step

## Directory Structure

```
learn-gpt2/
├── problems/
│   ├── 01-embeddings/
│   │   ├── README.md           # Learning objectives, concepts, hints
│   │   ├── problem.py          # Skeleton code with TODOs
│   │   ├── solution.py         # Complete, working solution
│   │   ├── test_embeddings.py  # Unit tests to validate solution
│   │   └── notebook.ipynb      # Interactive playground
│   ├── 02-attention-basics/
│   │   ├── README.md
│   │   ├── problem.py
│   │   ├── solution.py
│   │   ├── test_attention.py
│   │   └── notebook.ipynb
│   ├── 03-scaled-dot-product/
│   ├── 04-multi-head-attention/
│   ├── 05-causal-masking/
│   ├── 06-feedforward-network/
│   ├── 07-layer-norm-residuals/
│   ├── 08-transformer-block/
│   ├── 09-gpt2-config/
│   ├── 10-full-model/
│   ├── 11-weight-initialization/
│   ├── 12-pretrained-loading/
│   └── README.md               # Overview of all problems
│
└── src/gpt2/model.py           # Reference implementation (used for solutions)
```

## Learning Path: 12 Progressive Problems

### Problem 1: Token & Position Embeddings
**Difficulty**: ⭐ Easy
**Time estimate**: 30-45 minutes

**Learning Objectives**:
- Understand what embeddings are and why we need them
- Learn about token embeddings (vocabulary → dense vectors)
- Learn about position embeddings (sequence position → dense vectors)
- Understand embedding combination and dropout

**What to Implement**:
```python
class Embeddings(nn.Module):
    """Token and position embeddings for GPT-2."""

    def __init__(self, vocab_size, n_positions, n_embd, dropout):
        # TODO: Create token embedding layer (vocab_size → n_embd)
        # TODO: Create position embedding layer (n_positions → n_embd)
        # TODO: Create dropout layer
        pass

    def forward(self, input_ids):
        # TODO: Get token embeddings from input_ids
        # TODO: Create position indices [0, 1, 2, ..., seq_len-1]
        # TODO: Get position embeddings
        # TODO: Combine token + position embeddings
        # TODO: Apply dropout
        # TODO: Return combined embeddings
        pass
```

**Tests**:
- Output shape is `(batch_size, seq_len, n_embd)`
- Different input tokens produce different embeddings
- Position information is correctly added
- Dropout is applied in training mode

**Hints**:
💡 **Getting Started**
- Use `nn.Embedding(num_embeddings, embedding_dim)` for both token and position embeddings
- Token embeddings map vocabulary indices to vectors, position embeddings map positions to vectors

💡 **Implementation Tips**
- Create position indices using `torch.arange(0, seq_len)` on the same device as input_ids
- Expand position indices to match batch size: `position_ids.unsqueeze(0).expand(batch_size, seq_len)`
- Combine embeddings with simple addition: `token_emb + position_emb`
- Apply dropout after combining embeddings

💡 **Common Pitfalls**
- Don't forget to put position indices on the correct device (use `input_ids.device`)
- Position embeddings should have shape `(n_positions, n_embd)`, not `(seq_len, n_embd)`
- Make sure batch dimensions align when combining embeddings

💡 **Testing Tips**
- Print shapes at each step to verify dimensions
- Check that changing input tokens changes the output
- Verify dropout is only active in training mode: use `model.train()` and `model.eval()`

**Notebook Activities**:
- Visualize embedding vectors for different tokens
- Plot position embeddings to see patterns
- Experiment with different embedding dimensions
- Compare random vs learned embeddings

**Resources**:

📚 **PyTorch Documentation**
- [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) - Embedding layer documentation
- [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html) - Create position indices
- [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - Dropout layer
- [Tensor.unsqueeze](https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html) - Add dimensions to tensors
- [Tensor.expand](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) - Expand tensor dimensions

📄 **Papers & Articles**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Section 3.4 (Embeddings and Softmax)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Understanding embeddings visually

💻 **Additional Resources**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation of embeddings in transformers
- [Understanding Word Embeddings](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)

---

### Problem 2: Attention Basics
**Difficulty**: ⭐⭐ Medium
**Time estimate**: 45-60 minutes

**Learning Objectives**:
- Understand the Query-Key-Value attention mechanism
- Learn how attention allows tokens to "look at" other tokens
- Implement basic single-head attention (without scaling)

**What to Implement**:
```python
class SimpleAttention(nn.Module):
    """Basic single-head attention without scaling."""

    def __init__(self, n_embd):
        # TODO: Create Q, K, V linear projections
        pass

    def forward(self, x):
        # TODO: Compute Q, K, V from input
        # TODO: Compute attention scores (Q @ K^T)
        # TODO: Apply softmax to get attention weights
        # TODO: Apply attention to values (attn_weights @ V)
        # TODO: Return output
        pass
```

**Tests**:
- Q, K, V have correct shapes
- Attention weights sum to 1
- Output shape matches input shape
- Attention is position-aware

**Notebook Activities**:
- Visualize attention patterns as heatmaps
- See which tokens attend to which
- Experiment with different sequence lengths
- Understand attention weight interpretation

---

### Problem 3: Scaled Dot-Product Attention
**Difficulty**: ⭐⭐ Medium
**Time estimate**: 30-45 minutes

**Learning Objectives**:
- Understand why scaling is necessary (numerical stability)
- Learn about the scaling factor (√d_k)
- Handle softmax numerical stability

**What to Implement**:
```python
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with proper normalization."""

    def __init__(self, n_embd, dropout=0.1):
        # TODO: Store dimension for scaling
        # TODO: Create attention dropout layer
        pass

    def forward(self, q, k, v):
        # TODO: Compute attention scores (Q @ K^T)
        # TODO: Scale by √d_k
        # TODO: Apply softmax
        # TODO: Apply dropout to attention weights
        # TODO: Compute output (attn_weights @ V)
        # TODO: Return output and attention weights
        pass
```

**Tests**:
- Scaling prevents gradient issues with large dimensions
- Dropout is applied correctly
- Numerical stability with large embedding dimensions
- Attention weights still sum to 1

**Notebook Activities**:
- Compare scaled vs unscaled attention
- Visualize gradient flow
- Test with different embedding dimensions
- Plot attention weight distributions

---

### Problem 4: Multi-Head Attention
**Difficulty**: ⭐⭐⭐ Hard
**Time estimate**: 60-90 minutes

**Learning Objectives**:
- Understand why multiple attention heads are beneficial
- Learn to split embeddings across heads
- Implement parallel attention computation
- Concatenate and project multi-head outputs

**What to Implement**:
```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, n_embd, n_head, dropout=0.1):
        # TODO: Validate n_embd is divisible by n_head
        # TODO: Calculate head_dim
        # TODO: Create combined Q,K,V projection (efficient)
        # TODO: Create output projection
        # TODO: Create dropout layers
        pass

    def forward(self, x):
        # TODO: Project to Q, K, V
        # TODO: Split into multiple heads
        # TODO: Reshape for parallel attention
        # TODO: Compute scaled dot-product attention per head
        # TODO: Concatenate heads
        # TODO: Apply output projection
        # TODO: Apply dropout
        # TODO: Return output
        pass
```

**Tests**:
- Correct head splitting and reshaping
- Each head computes independent attention
- Output shape matches input shape
- Equivalent to single-head when n_head=1

**Notebook Activities**:
- Visualize different heads attending to different patterns
- Compare 1-head vs 12-head attention
- Plot head specialization
- Experiment with different head counts

---

### Problem 5: Causal Masking
**Difficulty**: ⭐⭐ Medium
**Time estimate**: 45-60 minutes

**Learning Objectives**:
- Understand autoregressive language modeling
- Learn why we prevent attending to future tokens
- Implement triangular causal mask
- Handle masking in attention computation

**What to Implement**:
```python
class CausalMultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking."""

    def __init__(self, n_embd, n_head, n_positions, dropout=0.1):
        # TODO: Initialize multi-head attention components
        # TODO: Register causal mask buffer (lower triangular)
        pass

    def forward(self, x):
        # TODO: Compute Q, K, V and split into heads
        # TODO: Compute attention scores
        # TODO: Apply causal mask (set future positions to -inf)
        # TODO: Apply softmax (masked positions become 0)
        # TODO: Apply attention to values
        # TODO: Concatenate and project
        # TODO: Return output
        pass
```

**Tests**:
- Future tokens don't influence past tokens
- Mask creates triangular attention pattern
- Masking doesn't affect valid positions
- Works with variable sequence lengths

**Notebook Activities**:
- Visualize causal attention patterns
- Compare causal vs bidirectional attention
- Test autoregressive generation
- See masked vs unmasked attention heatmaps

---

### Problem 6: Feedforward Network (MLP)
**Difficulty**: ⭐ Easy
**Time estimate**: 30 minutes

**Learning Objectives**:
- Understand the role of FFN in transformers
- Learn about expansion factor (4x hidden size)
- Implement GELU activation function
- Apply dropout for regularization

**What to Implement**:
```python
class NewGELU(nn.Module):
    """GELU activation (GPT-2 approximation)."""

    def forward(self, x):
        # TODO: Implement GELU approximation
        # Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        pass

class FeedForward(nn.Module):
    """Position-wise feedforward network."""

    def __init__(self, n_embd, n_inner, dropout=0.1):
        # TODO: Create expansion layer (n_embd → n_inner)
        # TODO: Create projection layer (n_inner → n_embd)
        # TODO: Create GELU activation
        # TODO: Create dropout
        pass

    def forward(self, x):
        # TODO: Expand dimension
        # TODO: Apply GELU
        # TODO: Project back
        # TODO: Apply dropout
        # TODO: Return output
        pass
```

**Tests**:
- Correct expansion to 4x hidden dimension
- GELU activation matches expected values
- Output shape matches input shape
- Dropout applied correctly

**Notebook Activities**:
- Compare GELU vs ReLU activations
- Visualize activation functions
- Test different expansion factors
- Plot intermediate representations

---

### Problem 7: Layer Normalization & Residuals
**Difficulty**: ⭐⭐ Medium
**Time estimate**: 45 minutes

**Learning Objectives**:
- Understand layer normalization
- Learn about residual connections and gradient flow
- Implement pre-norm architecture
- Understand normalization statistics

**What to Implement**:
```python
class ResidualBlock(nn.Module):
    """Block with layer norm and residual connection."""

    def __init__(self, n_embd, layer_norm_epsilon=1e-5):
        # TODO: Create layer normalization
        pass

    def forward(self, x, sublayer):
        # TODO: Apply layer norm (pre-norm)
        # TODO: Apply sublayer (attention or FFN)
        # TODO: Add residual connection
        # TODO: Return output
        pass
```

**Tests**:
- LayerNorm normalizes across embedding dimension
- Mean ≈ 0, variance ≈ 1 after normalization
- Residual connections preserve gradient flow
- Pre-norm vs post-norm behavior

**Notebook Activities**:
- Visualize normalization statistics
- Compare pre-norm vs post-norm architectures
- Plot gradient flow with/without residuals
- Test deep network stability

---

### Problem 8: Complete Transformer Block
**Difficulty**: ⭐⭐⭐ Hard
**Time estimate**: 60-90 minutes

**Learning Objectives**:
- Combine all previous components
- Implement full transformer block
- Understand information flow
- Connect attention and feedforward paths

**What to Implement**:
```python
class TransformerBlock(nn.Module):
    """Complete transformer block: Attention + FFN with residuals."""

    def __init__(self, n_embd, n_head, n_positions, dropout=0.1):
        # TODO: Create layer norm 1
        # TODO: Create causal multi-head attention
        # TODO: Create layer norm 2
        # TODO: Create feedforward network
        pass

    def forward(self, x):
        # TODO: Attention block: x = x + attn(ln(x))
        # TODO: Feedforward block: x = x + ffn(ln(x))
        # TODO: Return output
        pass
```

**Tests**:
- Full forward pass works
- Gradients flow through entire block
- Shapes preserved throughout
- Causal masking maintained

**Notebook Activities**:
- Visualize activations at each stage
- Track tensor shapes through the block
- Plot attention + FFN contributions
- Compare single vs stacked blocks

---

### Problem 9: GPT-2 Configuration
**Difficulty**: ⭐ Easy
**Time estimate**: 20-30 minutes

**Learning Objectives**:
- Organize model hyperparameters
- Support different model sizes
- Validate configuration constraints
- Create reusable config objects

**What to Implement**:
```python
class GPT2Config:
    """Configuration for GPT-2 model."""

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
    ):
        # TODO: Store all configuration parameters
        # TODO: Validate n_embd divisible by n_head
        # TODO: Compute n_inner (4 * n_embd)
        pass

    @classmethod
    def gpt2_small(cls):
        """Returns config for GPT-2 small (124M parameters)."""
        return cls()

    @classmethod
    def gpt2_medium(cls):
        """Returns config for GPT-2 medium (355M parameters)."""
        return cls(n_embd=1024, n_layer=24, n_head=16)

    # TODO: Add gpt2_large and gpt2_xl configs
```

**Tests**:
- All parameters stored correctly
- Validation catches invalid configs
- Preset configs match official sizes
- Computed values correct

**Notebook Activities**:
- Compare different model sizes
- Calculate parameter counts
- Visualize size vs performance tradeoffs
- Create custom configs

---

### Problem 10: Full GPT-2 Model Assembly
**Difficulty**: ⭐⭐⭐ Hard
**Time estimate**: 90-120 minutes

**Learning Objectives**:
- Assemble all components into complete model
- Stack transformer blocks
- Implement language modeling head
- Handle weight tying

**What to Implement**:
```python
class GPT2Model(nn.Module):
    """Complete GPT-2 language model."""

    def __init__(self, config):
        # TODO: Store config
        # TODO: Create token embeddings
        # TODO: Create position embeddings
        # TODO: Create embedding dropout
        # TODO: Create stack of transformer blocks
        # TODO: Create final layer norm
        # TODO: Create language model head
        # TODO: Tie weights (wte.weight = lm_head.weight)
        pass

    def forward(self, input_ids):
        # TODO: Get embeddings (token + position)
        # TODO: Apply dropout
        # TODO: Pass through transformer blocks
        # TODO: Apply final layer norm
        # TODO: Project to vocabulary (logits)
        # TODO: Return logits
        pass
```

**Tests**:
- End-to-end forward pass
- Correct output shape (batch, seq_len, vocab_size)
- Handles different sequence lengths
- Weight tying works correctly
- Gradients flow through entire model

**Notebook Activities**:
- Test with random inputs
- Visualize model structure
- Calculate total parameters
- Generate simple predictions
- Plot logit distributions

---

### Problem 11: Weight Initialization
**Difficulty**: ⭐⭐ Medium
**Time estimate**: 45 minutes

**Learning Objectives**:
- Understand importance of initialization
- Learn standard initialization schemes
- Implement layer-specific initialization
- Ensure training stability

**What to Implement**:
```python
class GPT2Model(nn.Module):
    # ... (previous code)

    def _init_weights(self, module):
        """Initialize weights for different layer types."""
        # TODO: For Linear layers:
        #   - Initialize weight with normal(0, 0.02)
        #   - Initialize bias to zeros
        # TODO: For Embedding layers:
        #   - Initialize weight with normal(0, 0.02)
        # TODO: For LayerNorm:
        #   - Initialize weight to ones
        #   - Initialize bias to zeros
        pass

    def __init__(self, config):
        # ... (previous initialization code)
        # TODO: Apply initialization to all modules
        self.apply(self._init_weights)
```

**Tests**:
- Weight statistics match expected distribution
- Biases initialized to zero
- LayerNorm initialized correctly
- No NaN or inf values after initialization
- Forward pass stable with random init

**Notebook Activities**:
- Plot weight distributions
- Compare different initialization schemes
- Test training stability
- Visualize activation statistics

---

### Problem 12: Loading Pretrained Weights
**Difficulty**: ⭐⭐⭐⭐ Very Hard
**Time estimate**: 120-180 minutes

**Learning Objectives**:
- Load weights from HuggingFace transformers
- Handle architecture differences (Conv1D vs Linear)
- Map state dict keys correctly
- Validate implementation correctness

**What to Implement**:
```python
class GPT2Model(nn.Module):
    # ... (previous code)

    @staticmethod
    def _build_weight_mapping(n_layers):
        """Build mapping from HF keys to our keys."""
        # TODO: Map embedding weights
        # TODO: Map layer norm weights
        # TODO: For each layer, map:
        #   - Attention weights (handle Conv1D transpose)
        #   - MLP weights (handle Conv1D transpose)
        #   - Layer norms
        # TODO: Return complete mapping dict
        pass

    @classmethod
    def from_pretrained(cls, model_name="gpt2"):
        """Load pretrained GPT-2 from HuggingFace."""
        # TODO: Import transformers
        # TODO: Load HF model and config
        # TODO: Create our config from HF config
        # TODO: Initialize our model
        # TODO: Build weight mapping
        # TODO: Copy weights (transpose Conv1D layers)
        # TODO: Load state dict
        # TODO: Return model
        pass
```

**Tests**:
- Successfully loads pretrained weights
- Output matches HuggingFace model (within tolerance)
- All weights copied correctly
- Conv1D transposition handled properly
- Works for all model sizes (small, medium, large, xl)

**Notebook Activities**:
- Compare outputs with HuggingFace
- Generate text with pretrained model
- Visualize learned embeddings
- Test on real prompts
- Analyze attention patterns

---

## File Templates

### README.md Template (per problem)
```markdown
# Problem [N]: [Title]

## Learning Objectives
- Objective 1
- Objective 2
- Objective 3

## Background
[Conceptual explanation of what you're implementing]

## Your Task
[Clear description of what to implement]

## Hints
💡 **Getting Started**
- Hint for getting started with the problem

💡 **Implementation Tips**
- Specific implementation hint 1
- Specific implementation hint 2

💡 **Common Pitfalls**
- Warning about common mistake 1
- Warning about common mistake 2

💡 **Testing Tips**
- How to verify your implementation is correct

## Testing Your Solution
```bash
python -m pytest test_[problem].py -v
```

## Resources

📚 **PyTorch Documentation**
- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) - Base class for all neural network modules
- [Relevant PyTorch function 1]
- [Relevant PyTorch function 2]

📄 **Papers & Articles**
- [Link to relevant paper section with specific page/section numbers]
- [Link to blog post or tutorial]

🎥 **Video Resources** (Optional)
- [Link to relevant video explanation]

💻 **Additional Tutorials**
- [Link to related tutorial or guide]

## Next Steps
Once you pass all tests, move on to Problem [N+1]!
```

### problem.py Template
```python
"""Problem [N]: [Title]

Learning objectives:
- Objective 1
- Objective 2

TODO: Implement the [component] class below.
"""

import torch
import torch.nn as nn
from torch import Tensor


class [Component](nn.Module):
    """[Brief description]."""

    def __init__(self, ...):
        """Initialize [component].

        Args:
            ...: Description

        Hints:
            - Use nn.Linear for ...
            - Don't forget to ...
        """
        super().__init__()
        # TODO: Implement initialization
        raise NotImplementedError("Complete the __init__ method")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)

        Hints:
            - Step 1: ...
            - Step 2: ...
            - Step 3: ...
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Complete the forward method")
```

### test_*.py Template
```python
"""Tests for Problem [N]: [Title]."""

import pytest
import torch
from problem import [Component]


def test_initialization():
    """Test that component initializes without errors."""
    component = [Component](...)
    assert component is not None


def test_forward_shape():
    """Test that forward pass produces correct output shape."""
    component = [Component](...)
    x = torch.randn(2, 10, 768)  # batch_size=2, seq_len=10, n_embd=768
    output = component(x)
    assert output.shape == (2, 10, 768)


def test_specific_behavior():
    """Test specific expected behavior."""
    # TODO: Add specific tests for this component
    pass


# Add more tests as needed
```

### notebook.ipynb Template
```python
# Cell 1: Setup
import sys
sys.path.append(".")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from problem import [Component]

# Cell 2: Create instance
component = [Component](...)
print(component)

# Cell 3: Test with sample data
x = torch.randn(1, 10, 768)
output = component(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Cell 4: Visualization
# TODO: Add visualization specific to this component

# Cell 5: Experiments
# TODO: Add interactive experiments
```

## Solutions Structure

Each solution file should:
1. Contain complete, working implementation
2. Include detailed comments explaining design choices
3. Follow best practices
4. Match or improve upon the reference implementation in `src/gpt2/model.py`

Solutions will be directly derived from `src/gpt2/model.py` with additional educational comments.

## Implementation Plan

### Phase 1: Infrastructure Setup
1. Create `problems/` and `solutions/` directory structure
2. Create main README files explaining the learning path
3. Set up pytest configuration for running tests

### Phase 2: Problem Creation (Problems 1-6)
1. For each problem:
   - Write README.md with learning objectives and concepts
   - Create problem.py skeleton with TODOs and hints
   - Write comprehensive test_*.py file
   - Create interactive notebook.ipynb
   - Extract solution from src/gpt2/model.py into solutions/

### Phase 3: Problem Creation (Problems 7-12)
1. Continue with remaining problems following same pattern
2. Ensure progressive difficulty and building on previous solutions

### Phase 4: Integration & Documentation
1. Create main problems/README.md with learning path overview
2. Add solutions/README.md with usage guidelines
3. Update project README.md with learning path information
4. Add CI to run all problem tests

### Phase 5: Testing & Refinement
1. Verify all tests pass
2. Ensure notebooks run without errors
3. Get feedback from test users
4. Refine explanations and hints

## Success Criteria

A learner who completes all 12 problems should:
- ✅ Understand transformer architecture deeply
- ✅ Have implemented GPT-2 from scratch
- ✅ Be able to load and use pretrained weights
- ✅ Understand attention mechanisms thoroughly
- ✅ Have practical PyTorch implementation experience
- ✅ Be prepared to implement other transformer variants
- ✅ Have working code that matches production implementations

## Dependencies

Required in each problem's virtual environment:
```toml
dependencies = [
    "torch>=2.0.0",
    "pytest>=7.0.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.0.0",
    "numpy>=1.20.0",
]

# Problem 12 additionally requires:
[problems.12]
dependencies = [
    "transformers>=4.0.0",
]
```

## Maintenance

- Keep solutions synchronized with `src/gpt2/model.py`
- Update tests as PyTorch evolves
- Gather feedback from learners
- Continuously improve explanations
- Add more visualizations based on user feedback

## Future Enhancements

Potential additions beyond the initial 12 problems:
- Problem 13: Text Generation (greedy, sampling, top-k, top-p)
- Problem 14: KV Caching for efficient generation
- Problem 15: Fine-tuning on custom datasets
- Problem 16: LoRA for efficient fine-tuning
- Problem 17: Quantization for deployment
- Problem 18: Distributed training

---

## Reference Resources by Problem

### Problem 2: Attention Basics
**PyTorch Functions:**
- `nn.Linear`, `torch.matmul` or `@`, `torch.softmax`, `Tensor.transpose`

**Key Concepts:**
- Query, Key, Value projections
- Attention weights computation
- Softmax normalization

**Papers:**
- Attention Is All You Need (Section 3.2.1)

### Problem 3: Scaled Dot-Product Attention
**PyTorch Functions:**
- `torch.matmul`, `torch.softmax`, `nn.Dropout`, `torch.sqrt`

**Key Concepts:**
- Scaling factor √d_k
- Numerical stability

**Papers:**
- Attention Is All You Need (Section 3.2.1, Equation 1)

### Problem 4: Multi-Head Attention
**PyTorch Functions:**
- `Tensor.view`, `Tensor.transpose`, `Tensor.contiguous`, `torch.cat` or `Tensor.reshape`

**Key Concepts:**
- Head splitting and concatenation
- Parallel attention computation
- Output projection

**Papers:**
- Attention Is All You Need (Section 3.2.2)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/#self-attention-in-detail)

### Problem 5: Causal Masking
**PyTorch Functions:**
- `torch.tril`, `torch.ones`, `register_buffer`, `Tensor.masked_fill`

**Key Concepts:**
- Autoregressive generation
- Causal (lower-triangular) mask
- Masking with -inf before softmax

**Papers:**
- Language Models are Unsupervised Multitask Learners (GPT-2 paper)
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html)

### Problem 6: Feedforward Network
**PyTorch Functions:**
- `nn.Linear`, `torch.tanh`, `torch.pow`, `nn.GELU` (for comparison)

**Key Concepts:**
- Position-wise feedforward
- GELU activation (approximate form)
- Expansion factor (4x)

**Papers:**
- Attention Is All You Need (Section 3.3)
- [GELU paper](https://arxiv.org/abs/1606.08415)
- [GPT-2 source code](https://github.com/openai/gpt-2/blob/master/src/model.py#L25-L26)

### Problem 7: Layer Normalization & Residuals
**PyTorch Functions:**
- `nn.LayerNorm`, addition operator for residuals

**Key Concepts:**
- Pre-norm vs post-norm
- Residual connections
- Gradient flow

**Papers:**
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Deep Residual Learning (ResNet paper)](https://arxiv.org/abs/1512.03385)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-norm analysis

### Problem 8: Transformer Block
**PyTorch Functions:**
- Combining previous components

**Key Concepts:**
- Information flow through blocks
- Pre-norm architecture pattern
- Residual paths

**Papers:**
- Attention Is All You Need (Section 3.1, Figure 1)
- Language Models are Unsupervised Multitask Learners

### Problem 9: GPT-2 Configuration
**PyTorch Functions:**
- Python dataclasses or regular classes

**Key Concepts:**
- Model sizes (small: 124M, medium: 355M, large: 774M, xl: 1.5B)
- Hyperparameter organization

**Papers:**
- [GPT-2 model card](https://github.com/openai/gpt-2/blob/master/model_card.md)

### Problem 10: Full GPT-2 Model
**PyTorch Functions:**
- `nn.ModuleList`, weight tying via assignment

**Key Concepts:**
- End-to-end architecture
- Weight tying between embeddings and LM head
- Stacking transformer blocks

**Papers:**
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) - Weight tying
- Language Models are Unsupervised Multitask Learners (full paper)

### Problem 11: Weight Initialization
**PyTorch Functions:**
- `nn.init.normal_`, `nn.init.zeros_`, `nn.init.ones_`, `Module.apply`

**Key Concepts:**
- Initialization schemes
- Training stability
- Standard deviation selection (0.02)

**Papers:**
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) - Xavier/Glorot init
- GPT-2 paper and code for specific initialization strategy

### Problem 12: Loading Pretrained Weights
**PyTorch Functions:**
- `state_dict()`, `load_state_dict()`, `Tensor.t()` for transpose, `Tensor.clone()`

**Key Concepts:**
- HuggingFace Conv1D vs PyTorch Linear
- Weight mapping
- State dict manipulation

**Documentation:**
- [HuggingFace Transformers GPT2Model](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [PyTorch State Dict Tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Conv1D explanation](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py)
