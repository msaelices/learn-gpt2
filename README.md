# Learn GPT-2

Repo to learn how to implement GPT-2 using Pytorch.

This project is a learning resource to understand the architecture and implementation of GPT-2. It is not intended for production use.

## Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd gpt2
uv sync --all-extras --dev
```

## Development

### Code Quality

Run linting with ruff:

```bash
uv run ruff check .
```

Format code with ruff:

```bash
uv run ruff format .
```

Run type checking with ty:

```bash
uvx ty check src
```

### Running Tests

Run all tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=gpt2 --cov-report=html
```

### Jupyter Notebooks

To explore the educational notebooks in the `notebooks/` directory:

```bash
uv run jupyter notebook notebooks/
```

This will start a Jupyter server and open your browser to browse the notebooks.
