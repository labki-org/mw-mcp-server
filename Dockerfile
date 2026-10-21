# Base image pinned by digest (python:3.12-slim). Bump deliberately:
#   docker buildx imagetools inspect python:3.12-slim
ARG PYTHON_IMAGE=python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

# ------------------------------------------------------------
# Builder Stage: resolve and install pinned dependencies
# ------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY requirements.lock ./

# Install the hash-checked lock into a separate prefix. All dependencies
# ship manylinux wheels, so no compiler toolchain is needed.
RUN pip install --require-hashes --only-binary=:all: --prefix=/install -r requirements.lock


# ------------------------------------------------------------
# Runtime Stage: Minimal production image
# ------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src"

# Create a lightweight non-root user
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src

# Copy Alembic migration files
COPY alembic.ini ./
COPY migrations/ ./migrations

# Data directory mounted as external volume in docker-compose
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Ensure non-root user runs the service
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Readiness probe (checks DB connectivity). curl is not in the slim image,
# so use the Python stdlib; any non-2xx or connection error exits non-zero.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
# Single worker by design: the embedding job queue is in-process memory and
# startup runs DB migrations, so multiple workers would split the queue and
# race on migrations. Scale with more replicas only after that changes.
CMD ["uvicorn", "mw_mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
