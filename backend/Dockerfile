# Backend Dockerfile for Real Estate AI Finder
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY backend/ ./backend/

# Create necessary directories
RUN mkdir -p data/raw data/cache/embeddings data/cache/vision

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

RUN echo "=== TAMAÑO DE CARPETAS ===" && \
du -h -d 1 /app && \
du -h -d 1 /app/src && \
du -h -d 1 /app/backend

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]