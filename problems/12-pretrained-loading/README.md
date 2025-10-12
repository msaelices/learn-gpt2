# Problem 12: Loading Pretrained Weights

## Learning Objectives
- Load pretrained GPT-2 weights from HuggingFace
- Handle architecture differences between implementations
- Map state dict keys correctly
- Deal with Conv1D vs Linear layer differences
- Validate implementation correctness against reference

## Background

You've implemented GPT-2 from scratch! But training a language model from scratch requires enormous compute resources (thousands of GPU hours). Fortunately, OpenAI released pretrained GPT-2 weights that we can load into our implementation.

However, there's a catch: HuggingFace's implementation uses slightly different layer names and types than our implementation. We need to:
1. **Map parameter names** from HuggingFace's naming scheme to ours
2. **Handle Conv1D layers** - HuggingFace uses `Conv1D` instead of `Linear`
3. **Transpose weights** where necessary
4. **Validate** that outputs match

### HuggingFace vs Our Implementation

**HuggingFace uses Conv1D for efficiency:**
```python
# HuggingFace:
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        self.weight = nn.Parameter(torch.empty(nx, nf))  # Note: (nx, nf)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

# Ours (standard PyTorch):
nn.Linear(nx, nf)  # weight shape: (nf, nx)
```

**Key difference**: Conv1D weight shape is `(nx, nf)`, Linear weight shape is `(nf, nx)`. We need to **transpose**!

### Parameter Name Mapping

HuggingFace uses different naming:

| HuggingFace | Our Implementation |
|-------------|-------------------|
| `transformer.wte.weight` | `wte.weight` |
| `transformer.wpe.weight` | `wpe.weight` |
| `transformer.h.{i}.ln_1.weight` | `h.{i}.ln_1.weight` |
| `transformer.h.{i}.attn.c_attn.weight` | `h.{i}.attn.c_attn.weight` |
| `transformer.h.{i}.attn.c_attn.bias` | `h.{i}.attn.c_attn.bias` |
| `transformer.h.{i}.attn.c_proj.weight` | `h.{i}.attn.c_proj.weight` |
| `transformer.h.{i}.attn.c_proj.bias` | `h.{i}.attn.c_proj.bias` |
| `transformer.h.{i}.ln_2.weight` | `h.{i}.ln_2.weight` |
| `transformer.h.{i}.mlp.c_fc.weight` | `h.{i}.mlp.c_fc.weight` |
| `transformer.h.{i}.mlp.c_fc.bias` | `h.{i}.mlp.c_fc.bias` |
| `transformer.h.{i}.mlp.c_proj.weight` | `h.{i}.mlp.c_proj.weight` |
| `transformer.h.{i}.mlp.c_proj.bias` | `h.{i}.mlp.c_proj.bias` |
| `transformer.ln_f.weight` | `ln_f.weight` |
| `transformer.ln_f.bias` | `ln_f.bias` |
| (lm_head tied with wte) | `lm_head.weight` (tied) |

## Your Task

Implement the `from_pretrained` class method that:
1. Loads a HuggingFace GPT-2 model
2. Creates our GPT2Config from HuggingFace's config
3. Initializes our model
4. Maps and copies weights (with transposition where needed)
5. Returns the loaded model

## Hints

💡 **Getting Started**
- Add `@classmethod` decorator to `from_pretrained` method
- Import `transformers` library: `from transformers import GPT2LMHeadModel`
- Load HuggingFace model: `hf_model = GPT2LMHeadModel.from_pretrained(model_name)`

💡 **Configuration Mapping**
```python
config = GPT2Config(
    vocab_size=hf_config.vocab_size,
    n_positions=hf_config.n_positions,
    n_embd=hf_config.n_embd,
    n_layer=hf_config.n_layer,
    n_head=hf_config.n_head,
    # ... etc
)
```

💡 **Weight Mapping Strategy**
- Build a mapping dictionary: `{hf_key: our_key}`
- Strip `transformer.` prefix from HuggingFace keys
- For each layer `i`, map all components
- Don't forget embeddings and final layer norm

💡 **Handling Conv1D Transposition**
```python
# HuggingFace Conv1D weights need transposition
if 'c_attn' in key or 'c_proj' in key or 'c_fc' in key:
    # These are Conv1D layers in HuggingFace
    param_data = hf_param.t()  # Transpose!
else:
    param_data = hf_param.clone()
```

💡 **Common Pitfalls**
- Don't forget to transpose Conv1D weights (c_attn, c_proj, c_fc)
- LayerNorm and embedding weights don't need transposition
- HuggingFace has `transformer.` prefix, we don't
- lm_head weights are tied with wte, don't copy separately

💡 **Testing Tips**
- Compare outputs with HuggingFace on same input
- Use `torch.allclose(our_output, hf_output, atol=1e-5)`
- Test with different model sizes (gpt2, gpt2-medium)
- Verify parameter count matches

## Testing Your Solution

```bash
cd problems/12-pretrained-loading
uv run pytest test_pretrained.py -v
```

## Resources

📚 **PyTorch Documentation**
- [torch.Tensor.t](https://pytorch.org/docs/stable/generated/torch.Tensor.t.html) - Transpose tensor
- [state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) - Get/load model parameters
- [load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict) - Load parameters

📄 **HuggingFace Documentation**
- [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel)
- [GPT2Config](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config)
- [from_pretrained](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained)

💻 **Additional Resources**
- [HuggingFace Conv1D](https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py) - Conv1D implementation
- [PyTorch State Dict Tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

## Key Concepts

**Conv1D vs Linear**

HuggingFace's Conv1D is mathematically equivalent to Linear but with transposed weights:

```python
# HuggingFace Conv1D:
# weight shape: (in_features, out_features)
# computation: x @ weight + bias

# PyTorch Linear:
# weight shape: (out_features, in_features)
# computation: x @ weight.t() + bias

# Solution: Transpose when loading!
our_linear.weight.data = hf_conv1d.weight.data.t()
```

**State Dict Structure**

```python
# HuggingFace state dict (simplified):
{
    'transformer.wte.weight': tensor(...),
    'transformer.h.0.attn.c_attn.weight': tensor(...),
    'transformer.h.0.attn.c_attn.bias': tensor(...),
    ...
}

# Our state dict:
{
    'wte.weight': tensor(...),
    'h.0.attn.c_attn.weight': tensor(...),
    'h.0.attn.c_attn.bias': tensor(...),
    ...
}
```

**Weight Mapping Process**

```python
# 1. Get HuggingFace state dict
hf_state = hf_model.state_dict()

# 2. Create empty state dict for our model
our_state = {}

# 3. Map and copy each parameter
for hf_key, hf_param in hf_state.items():
    # Strip 'transformer.' prefix
    our_key = hf_key.replace('transformer.', '')

    # Transpose if Conv1D layer
    if needs_transpose(our_key):
        our_state[our_key] = hf_param.t()
    else:
        our_state[our_key] = hf_param.clone()

# 4. Load into our model
our_model.load_state_dict(our_state, strict=False)
```

## Implementation Strategy

### Step 1: Load HuggingFace Model

```python
from transformers import GPT2LMHeadModel

hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
hf_config = hf_model.config
```

### Step 2: Create Our Config

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

### Step 3: Map Weights

```python
# Get HuggingFace state dict
hf_state = hf_model.state_dict()

# Create mapping
our_state = {}

for hf_key in hf_state.keys():
    # Remove 'transformer.' prefix
    if hf_key.startswith('transformer.'):
        our_key = hf_key[len('transformer.'):]
    else:
        continue  # Skip lm_head, it's tied

    # Get parameter
    param = hf_state[hf_key]

    # Transpose Conv1D weights
    if any(name in our_key for name in ['c_attn.weight', 'c_proj.weight', 'c_fc.weight']):
        param = param.t()

    our_state[our_key] = param
```

### Step 4: Load and Validate

```python
# Load state dict
model.load_state_dict(our_state, strict=False)

# Validate: compare outputs
model.eval()
hf_model.eval()

with torch.no_grad():
    our_output = model(test_input)
    hf_output = hf_model(test_input).logits

assert torch.allclose(our_output, hf_output, atol=1e-5)
```

## Expected Behavior

```python
# Load pretrained GPT-2
model = GPT2Model.from_pretrained('gpt2')

# Should work with all sizes
model_small = GPT2Model.from_pretrained('gpt2')         # 124M
model_medium = GPT2Model.from_pretrained('gpt2-medium') # 355M
model_large = GPT2Model.from_pretrained('gpt2-large')   # 774M
model_xl = GPT2Model.from_pretrained('gpt2-xl')         # 1.5B

# Outputs should match HuggingFace
input_ids = torch.tensor([[15496, 995, 318]])  # "Hello World is"
our_logits = model(input_ids)
hf_logits = hf_model(input_ids).logits

assert torch.allclose(our_logits, hf_logits, atol=1e-5)
print("✓ Outputs match HuggingFace!")
```

## Validation Checklist

After implementing `from_pretrained`:

- ✓ Model loads without errors
- ✓ All weights are copied correctly
- ✓ Output logits match HuggingFace (within 1e-5)
- ✓ Works for all model sizes (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- ✓ Conv1D weights are properly transposed
- ✓ Weight tying is preserved

## Next Steps

Congratulations! You've completed all 12 problems and built GPT-2 from scratch! 🎉

You now have:
- ✅ Complete understanding of transformer architecture
- ✅ Working GPT-2 implementation
- ✅ Ability to load and use pretrained models
- ✅ Solid foundation for exploring other LLMs

**What's next?**
- Experiment with text generation (greedy, sampling, top-k, top-p)
- Implement KV caching for efficient generation
- Try fine-tuning on custom datasets
- Explore LoRA for efficient fine-tuning
- Study other architectures (GPT-3, LLaMA, etc.)
