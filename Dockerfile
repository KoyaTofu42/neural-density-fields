# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:26.02-py3

# Install uv using a multi-stage build pattern
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Use the system Python to ensure we use the baked-in NVIDIA PyTorch
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Copy the entire application code
COPY . .

# Optimize uv package handling with cache mounts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e .

# Set Python path so `src` can be imported
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python"]
