# pa development tasks

default:
    just ci

# Full CI pipeline: lint, type check, test
ci:
    just lint
    just typecheck
    just test

# Lint with ruff (format + check)
lint:
    uv run ruff format --check pa/ tests/
    uv run ruff check pa/ tests/

# Auto-fix lint issues
fix:
    uv run ruff format pa/ tests/
    uv run ruff check --fix pa/ tests/

# Type check with pyright
typecheck:
    uv run pyright pa/

# Run tests
test:
    uv run pytest tests/ -v

# Run tests with coverage
test-cov:
    uv run pytest tests/ -v --cov=pa --cov-report=term-missing

# Install all dependencies including dev
install:
    uv sync --all-extras

# Clean build artifacts
clean:
    rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .pyright_cache/ .mypy_cache/ htmlcov/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
