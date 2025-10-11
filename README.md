# gpt2

GPT2 implementations

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
