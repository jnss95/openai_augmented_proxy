# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src ./src

# Install dependencies and the package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Default config directory
ENV OPENAI_PROXY_CONFIG_DIR=/config

# Expose the default port
EXPOSE 8000

# Run the proxy server
CMD ["uv", "run", "openai-proxy", "--host", "0.0.0.0"]
