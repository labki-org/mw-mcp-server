# ------------------------------------------------------------
# Base Stage: Build environment for dependencies
# ------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install system dependencies for FAISS, BLAS, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY pyproject.toml requirements.txt* ./

# Install dependencies into a separate directory
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ------------------------------------------------------------
# Runtime Stage: Minimal production image
# ------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Create a lightweight non-root user
RUN useradd -m -u 1001 appuser

# Install only runtime dependencies for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src

ENV PYTHONPATH="/app/src"

# Data directory mounted as external volume in docker-compose
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Ensure non-root user runs the service
USER appuser

# Expose FastAPI port
EXPOSE 8000

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
CMD ["uvicorn", "mw_mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
