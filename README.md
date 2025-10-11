# gpt2
GPT2 implementations

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and Python environment management.

### Installation

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/msaelices/gpt2.git
cd gpt2
```

3. Install dependencies:
```bash
uv sync --dev
```

### Development

#### Running tests
```bash
uv run pytest
```

#### Running linter
```bash
# Check for linting issues
uv run ruff check src tests

# Fix linting issues automatically
uv run ruff check --fix src tests
```

#### Formatting code
```bash
# Check formatting
uv run ruff format --check src tests

# Format code
uv run ruff format src tests
```

### Project Structure
```
gpt2/
├── src/
│   └── gpt2/          # Main package
├── tests/             # Test files
├── pyproject.toml     # Project configuration
└── README.md          # This file
```
