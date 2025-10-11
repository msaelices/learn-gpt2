# gpt2

GPT-2 implementation for educational purposes only.

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

### Running Tests

Run all tests:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=gpt2 --cov-report=html
```
