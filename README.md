# Learn GPT-2: A Progressive Learning Path

A hands-on, problem-based learning curriculum for implementing GPT-2 from scratch using PyTorch. Learn transformer architecture through 12 progressive problems that build on each other.

## 🎯 What You'll Learn

By completing this learning path, you will:
- ✅ Understand transformer architecture deeply
- ✅ Implement GPT-2 from scratch in PyTorch
- ✅ Master attention mechanisms (self-attention, multi-head, causal masking)
- ✅ Learn layer normalization and residual connections
- ✅ Load and use pretrained GPT-2 weights from HuggingFace
- ✅ Gain practical PyTorch implementation experience

## 📚 Learning Path Overview

The curriculum consists of 12 progressive problems:

| # | Problem | Difficulty | Status |
|---|---------|------------|--------|
| 1 | [Token & Position Embeddings](problems/01-embeddings/) | ⭐ Easy | ✅ Complete |
| 2 | [Attention Basics](problems/02-attention-basics/) | ⭐⭐ Medium | ✅ Complete |
| 3 | [Scaled Dot-Product Attention](problems/03-scaled-attention/) | ⭐⭐ Medium | ✅ Complete |
| 4 | [Multi-Head Attention](problems/04-multi-head-attention/) | ⭐⭐⭐ Hard | ✅ Complete |
| 5 | [Causal Masking](problems/05-causal-masking/) | ⭐⭐ Medium | ✅ Complete |
| 6 | [Feedforward Network](problems/06-feedforward-network/) | ⭐ Easy | ✅ Complete |
| 7 | [Layer Normalization & Residuals](problems/07-layer-norm-residuals/) | ⭐⭐ Medium | ✅ Complete |
| 8 | [Complete Transformer Block](problems/08-transformer-block/) | ⭐⭐⭐ Hard | ✅ Complete |
| 9 | [GPT-2 Configuration](problems/09-gpt2-config/) | ⭐ Easy | ✅ Complete |
| 10 | [Full GPT-2 Model Assembly](problems/10-full-model/) | ⭐⭐⭐ Hard | ✅ Complete |
| 11 | [Weight Initialization](problems/11-weight-initialization/) | ⭐⭐ Medium | ✅ Complete |
| 12 | [Loading Pretrained Weights](problems/12-pretrained-loading/) | ⭐⭐⭐⭐ Very Hard | ✅ Complete |

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd learn-gpt2
uv sync --all-extras
```

This will install all dependencies including PyTorch, transformers, matplotlib, seaborn, jupyter, and pytest.

## 📖 How to Use This Learning Path

### Problem Structure

Each problem directory contains:

```
problems/XX-problem-name/
├── README.md              # Learning objectives, concepts, and hints
├── problem.py             # Skeleton code with TODOs for you to implement
├── solution.py            # Complete working solution (don't peek!)
├── test_*.py              # Unit tests to validate your implementation
└── notebook.ipynb         # Interactive notebook for exploration
```

### Learning Workflow

For each problem, follow this workflow:

#### 1️⃣ Read the Problem

```bash
# Navigate to the problem directory
cd problems/01-embeddings

# Read the README to understand concepts
cat README.md
```

The README contains:
- Learning objectives
- Background concepts
- Your task description
- Implementation hints
- Common pitfalls
- Resources and references

#### 2️⃣ Implement Your Solution

Edit `problem.py` and implement the TODOs:

Tips:
- Read all the hints in the docstrings
- Start with the `__init__` method, then `forward`
- Print shapes to debug: `print(f"Shape: {tensor.shape}")`
- Don't worry about getting it perfect - iterate!

#### 3️⃣ Test Your Implementation

Run the tests to validate your solution:

```bash
# Run tests for the current problem
uv run pytest test_*.py -v

# Run with more details
uv run pytest test_*.py -v -s
```

Tests will show you:
- ✅ What's working correctly
- ❌ What needs to be fixed
- Helpful error messages

#### 4️⃣ Explore Interactively

Use the Jupyter notebook to visualize and experiment:

```bash
# Start Jupyter (from project root)
uv run jupyter lab

# Or start from the problem directory
cd problems/01-embeddings
uv run jupyter notebook notebook.ipynb
```

**Important**: When the notebook opens, make sure to select the **"Python (learn-gpt2)"** kernel from the kernel selector in the top-right corner.

The notebooks include:
- Interactive visualizations
- Step-by-step execution
- Experimentation with different parameters
- Visual understanding of concepts

#### 5️⃣ Compare with Solution

If you get stuck or want to compare your implementation:

```bash
# View the solution (try not to peek too early!)
cat solution.py

# Or run tests against the solution to see expected behavior
uv run pytest test_*.py -v
```

#### 6️⃣ Move to Next Problem

Once all tests pass, move to the next problem:

```bash
cd ../02-attention-basics
```

Each problem builds on the previous ones, so it's important to complete them in order.

## 🧪 Running Tests

### Test Individual Problems

```bash
# Test a specific problem
cd problems/01-embeddings
uv run pytest test_embeddings.py -v

# Test with coverage
uv run pytest test_embeddings.py -v --cov=. --cov-report=term-missing
```

### Test All Problems

From the project root:

```bash
# Run all problem tests
uv run pytest problems/ -v

# Run tests in parallel (faster)
uv run pytest problems/ -v -n auto
```

### Test the Reference Implementation

```bash
# Test the complete reference implementation
uv run pytest tests/ -v
```

## 📓 Interactive Notebooks

All problems include interactive Jupyter notebooks for exploration and visualization.

### Starting Jupyter

```bash
# Start JupyterLab (recommended)
uv run jupyter lab

# Or start Jupyter Notebook
uv run jupyter notebook
```

### Using the Notebooks

1. Navigate to the problem directory (e.g., `problems/01-embeddings/`)
2. Open `notebook.ipynb`
3. **Select the "Python (learn-gpt2)" kernel** from the top-right corner
4. Run cells with `Shift+Enter`

The notebooks include:
- 📊 Visualizations of attention patterns, embeddings, and activations
- 🔬 Interactive experiments with different parameters
- 📈 Plots and heatmaps to understand what's happening
- ✅ Verification cells to test your implementation

## 🔧 Development Tools

### Code Formatting

Format code with ruff:

```bash
uv run ruff format .
```

### Linting

Check code quality:

```bash
uv run ruff check .
```

### Type Checking

Run type checking:

```bash
uvx ty check src
```

## 📚 Project Structure

```
learn-gpt2/
├── problems/              # Progressive learning problems
│   ├── 01-embeddings/
│   ├── 02-attention-basics/
│   ├── ...
│   └── 12-pretrained-loading/
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## 🎓 Learning Tips

### For Beginners

1. **Don't skip problems** - Each builds on previous concepts
2. **Read the hints** - Docstrings contain valuable implementation tips
3. **Start simple** - Get basic functionality working, then refine
4. **Use the notebooks** - Visual feedback helps understanding
5. **Compare with tests** - Tests show expected behavior

### For Intermediate Learners

1. **Try before peeking** - Attempt implementation before checking solution
2. **Experiment** - Modify parameters, try different approaches
3. **Read the papers** - Links provided in each README
4. **Optimize** - Think about efficiency and best practices

### For Advanced Learners

1. **Implement variations** - Try different architectures (post-norm, etc.)
2. **Profile performance** - Use PyTorch profiler to optimize
3. **Add features** - Implement additional functionality
4. **Contribute** - Submit improvements or additional problems

## 🐛 Troubleshooting

### Import Errors in Notebooks

If you see `ModuleNotFoundError` in Jupyter notebooks:

1. Make sure you've run `uv sync --all-extras`
2. **Select the correct kernel**: "Python (learn-gpt2)" in the top-right corner of the notebook
3. Restart the kernel: Kernel → Restart Kernel

### Tests Failing

- Make sure you've implemented all TODOs in `problem.py`
- Check that tensor shapes match expected dimensions
- Print intermediate shapes: `print(f"x.shape: {x.shape}")`
- Read the test error messages carefully - they often point to the issue

### uv Command Not Found

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal.

## 📖 Additional Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper

### Tutorials
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation
- [The Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html) - GPT-2 walkthrough

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## 🤝 Contributing

Contributions are welcome! Whether it's:
- Fixing typos or improving explanations
- Adding more tests or examples
- Creating new problems or variations
- Improving visualizations in notebooks

Please open an issue or submit a pull request.

## 📄 License

This project is a learning resource and is not intended for production use.

## 🙏 Acknowledgments

- Andrej Karpathy for his inspiring [YouTube channel](https://www.youtube.com/@andrejkarpathy) and "Neural Networks: Zero to Hero" course
- OpenAI for GPT-2
- HuggingFace for the transformers library
- The PyTorch team
- All the excellent transformer tutorials and papers that made this possible

---

Happy learning! 🚀 If you find this helpful, please give it a ⭐!
