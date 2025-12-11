# Production Dockerfile for YouTube Sentiment Analysis API
# Optimized for CPU inference with option for GPU

# Base image with Python 3.11 (stable for transformers)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY phase5_production_deployment/ ./phase5_production_deployment/
COPY config.yaml .
COPY models/ ./models/
COPY utils/ ./utils/

# Copy trained DistilBERT checkpoint (required for inference)
COPY phase3_transformer_models/checkpoints/distilbert/ ./phase3_transformer_models/checkpoints/distilbert/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "phase5_production_deployment.01_fastapi_server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
