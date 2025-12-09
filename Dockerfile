FROM python:3.11-slim

# Install system deps (e.g., for Faiss)
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

ENV PYTHONPATH=/app/src

# Expose FastAPI port
EXPOSE 8000

# Use uvicorn as entrypoint
CMD ["uvicorn", "mw_mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
