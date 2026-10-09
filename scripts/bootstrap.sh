#!/usr/bin/env bash
set -euo pipefail
uv lock
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
