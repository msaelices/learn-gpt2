# Specification: `from_pretrained` Method for GPT2Model

## Overview

Implement a class method `from_pretrained` in the `GPT2Model` class that loads pretrained weights from Hugging Face's GPT-2 models and maps them to our custom implementation.

## Motivation

- Enable users to load pretrained GPT-2 weights and use our educational implementation
- Support fine-tuning and experimentation with official pretrained models
- Validate that our implementation is architecturally compatible with official GPT-2
- Provide an excellent learning tool for understanding model weight transfer between frameworks

## Technical Design

### 1. Add Dependencies

**File: `pyproject.toml`**

Add `transformers` to the main dependencies:

```toml
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.0.0",
]
```

### 2. Implement `from_pretrained` Class Method

**File: `src/gpt2/model.py`**

Add to the `GPT2Model` class:

```python
@classmethod
def from_pretrained(cls, model_name: str = "gpt2") -> "GPT2Model":
    """Load pretrained GPT-2 weights from Hugging Face.

    Args:
        model_name: Name of the pretrained model (e.g., "gpt2", "gpt2-medium",
                   "gpt2-large", "gpt2-xl")

    Returns:
        GPT2Model instance with loaded pretrained weights.
    """
```

### 3. Implementation Steps

#### Step 1: Load Hugging Face Model
```python
from transformers import GPT2LMHeadModel

hf_model = GPT2LMHeadModel.from_pretrained(model_name)
hf_config = hf_model.config
```

#### Step 2: Create Compatible Configuration
```python
config = GPT2Config(
    vocab_size=hf_config.vocab_size,
    n_positions=hf_config.n_positions,
    n_embd=hf_config.n_embd,
    n_layer=hf_config.n_layer,
    n_head=hf_config.n_head,
    resid_pdrop=hf_config.resid_pdrop,
    embd_pdrop=hf_config.embd_pdrop,
    attn_pdrop=hf_config.attn_pdrop,
    layer_norm_epsilon=hf_config.layer_norm_epsilon,
)
```

#### Step 3: Initialize Our Model
```python
model = cls(config)
```

#### Step 4: Create Weight Mapping
Build a mapping dictionary that maps Hugging Face state dict keys to our model's keys:

```python
def _build_weight_mapping(n_layers: int) -> dict[str, str]:
    """Build mapping from HF model keys to our model keys.

    Args:
        n_layers: Number of transformer layers

    Returns:
        Dictionary mapping HF keys to our keys
    """
    mapping = {
        'transformer.wte.weight': 'wte.weight',
        'transformer.wpe.weight': 'wpe.weight',
        'transformer.ln_f.weight': 'ln_f.weight',
        'transformer.ln_f.bias': 'ln_f.bias',
    }

    # Add mappings for each transformer layer
    for i in range(n_layers):
        layer_mapping = {
            # Layer norm 1
            f'transformer.h.{i}.ln_1.weight': f'h.{i}.ln_1.weight',
            f'transformer.h.{i}.ln_1.bias': f'h.{i}.ln_1.bias',

            # Attention
            f'transformer.h.{i}.attn.c_attn.weight': f'h.{i}.attn.c_attn.weight',
            f'transformer.h.{i}.attn.c_attn.bias': f'h.{i}.attn.c_attn.bias',
            f'transformer.h.{i}.attn.c_proj.weight': f'h.{i}.attn.c_proj.weight',
            f'transformer.h.{i}.attn.c_proj.bias': f'h.{i}.attn.c_proj.bias',

            # Layer norm 2
            f'transformer.h.{i}.ln_2.weight': f'h.{i}.ln_2.weight',
            f'transformer.h.{i}.ln_2.bias': f'h.{i}.ln_2.bias',

            # MLP
            f'transformer.h.{i}.mlp.c_fc.weight': f'h.{i}.mlp.c_fc.weight',
            f'transformer.h.{i}.mlp.c_fc.bias': f'h.{i}.mlp.c_fc.bias',
            f'transformer.h.{i}.mlp.c_proj.weight': f'h.{i}.mlp.c_proj.weight',
            f'transformer.h.{i}.mlp.c_proj.bias': f'h.{i}.mlp.c_proj.bias',
        }
        mapping.update(layer_mapping)

    return mapping
```

#### Step 5: Copy Weights with Proper Handling

```python
# Get state dicts
hf_state_dict = hf_model.state_dict()
our_state_dict = model.state_dict()

# Build mapping
mapping = _build_weight_mapping(config.n_layer)

# Copy weights
for hf_key, our_key in mapping.items():
    if hf_key in hf_state_dict:
        weight = hf_state_dict[hf_key]

        # HF uses Conv1D which stores weights transposed compared to Linear
        # Conv1D weight shape: (in_features, out_features)
        # Linear weight shape: (out_features, in_features)
        if 'weight' in hf_key and weight.dim() == 2:
            # Check if this is a Conv1D layer (attn, mlp)
            if any(x in hf_key for x in ['c_attn', 'c_proj', 'c_fc']):
                weight = weight.t()  # Transpose for Conv1D -> Linear

        our_state_dict[our_key].copy_(weight)

# Load the modified state dict
model.load_state_dict(our_state_dict)
```

### 4. Weight Mapping Details

#### Hugging Face Model Structure
```
transformer.wte.weight               [50257, 768]
transformer.wpe.weight               [1024, 768]
transformer.h.{i}.ln_1.weight        [768]
transformer.h.{i}.ln_1.bias          [768]
transformer.h.{i}.attn.c_attn.weight [768, 2304]  # Conv1D - needs transpose
transformer.h.{i}.attn.c_attn.bias   [2304]
transformer.h.{i}.attn.c_proj.weight [768, 768]   # Conv1D - needs transpose
transformer.h.{i}.attn.c_proj.bias   [768]
transformer.h.{i}.ln_2.weight        [768]
transformer.h.{i}.ln_2.bias          [768]
transformer.h.{i}.mlp.c_fc.weight    [768, 3072]  # Conv1D - needs transpose
transformer.h.{i}.mlp.c_fc.bias      [3072]
transformer.h.{i}.mlp.c_proj.weight  [3072, 768]  # Conv1D - needs transpose
transformer.h.{i}.mlp.c_proj.bias    [768]
transformer.ln_f.weight              [768]
transformer.ln_f.bias                [768]
lm_head.weight                       [50257, 768]  # Tied with wte.weight
```

#### Our Model Structure
```
wte.weight           [50257, 768]
wpe.weight           [1024, 768]
h.{i}.ln_1.weight    [768]
h.{i}.ln_1.bias      [768]
h.{i}.attn.c_attn.weight [3*768, 768]  # Linear
h.{i}.attn.c_attn.bias   [2304]
h.{i}.attn.c_proj.weight [768, 768]    # Linear
h.{i}.attn.c_proj.bias   [768]
h.{i}.ln_2.weight    [768]
h.{i}.ln_2.bias      [768]
h.{i}.mlp.c_fc.weight    [3072, 768]   # Linear
h.{i}.mlp.c_fc.bias      [3072]
h.{i}.mlp.c_proj.weight  [768, 3072]   # Linear
h.{i}.mlp.c_proj.bias    [768]
ln_f.weight          [768]
ln_f.bias            [768]
lm_head.weight       [50257, 768]  # Tied with wte.weight
```

### 5. Testing

**File: `tests/test_from_pretrained.py`**

Create comprehensive tests:

```python
def test_from_pretrained_loads():
    """Test that from_pretrained loads successfully."""
    model = GPT2Model.from_pretrained("gpt2")
    assert model is not None
    assert isinstance(model, GPT2Model)

def test_from_pretrained_config():
    """Test that configuration matches HF model."""
    model = GPT2Model.from_pretrained("gpt2")
    assert model.config.vocab_size == 50257
    assert model.config.n_embd == 768
    assert model.config.n_layer == 12
    assert model.config.n_head == 12

def test_from_pretrained_forward():
    """Test that forward pass works with loaded weights."""
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    input_ids = torch.randint(0, 50257, (1, 10))
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 10, 50257)

def test_from_pretrained_matches_hf():
    """Test that output matches HF model (within tolerance)."""
    from transformers import GPT2LMHeadModel

    # Load both models
    our_model = GPT2Model.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set to eval mode
    our_model.eval()
    hf_model.eval()

    # Test with same input
    input_ids = torch.randint(0, 50257, (1, 10))

    with torch.no_grad():
        our_logits = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    # Check shapes match
    assert our_logits.shape == hf_logits.shape

    # Check values are close (allowing for numerical differences)
    assert torch.allclose(our_logits, hf_logits, atol=1e-4, rtol=1e-3)
```

## Implementation Checklist

- [ ] Add `transformers>=4.0.0` to `pyproject.toml` dependencies
- [ ] Implement `_build_weight_mapping` helper function
- [ ] Implement `from_pretrained` class method in `GPT2Model`
- [ ] Handle Conv1D to Linear weight transposition correctly
- [ ] Ensure weight tying between `wte.weight` and `lm_head.weight`
- [ ] Create `tests/test_from_pretrained.py` with comprehensive tests
- [ ] Update documentation/README with usage examples
- [ ] Verify all tests pass with loaded pretrained weights

## Usage Example

```python
from gpt2 import GPT2Model
import torch

# Load pretrained GPT-2 model
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# Generate predictions
input_ids = torch.randint(0, 50257, (1, 10))
with torch.no_grad():
    logits = model(input_ids)

# Get next token predictions
next_token_logits = logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1)
```

## Benefits

1. **Educational Value**: Users can experiment with pretrained weights while learning the architecture
2. **Validation**: Proves our implementation matches the official GPT-2 architecture
3. **Practical Use**: Enables fine-tuning and transfer learning experiments
4. **Debugging**: Helps identify any architectural discrepancies by comparing outputs
5. **Community Standard**: Follows the common pattern established by Hugging Face

## Notes

- The Conv1D layer in Hugging Face stores weights in `(in_features, out_features)` format, which is transposed compared to PyTorch's Linear layer `(out_features, in_features)`
- Weight tying between embeddings and output head is handled automatically since we set `self.lm_head.weight = self.wte.weight` in our model
- The `lm_head.weight` from HF should not be loaded separately as it's tied to `wte.weight`
