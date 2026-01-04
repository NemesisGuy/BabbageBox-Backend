# Babbagebox-Backend Dockerfile (no-AVX for older CPUs)
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgomp1 \
    libopenblas-dev \
    libsndfile1 \
    libsndfile1-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Disable AVX so llama-cpp-python builds for older CPUs
    CMAKE_ARGS="-DLLAMA_NATIVE=OFF" \
    LLAMA_CUBLAS=0 \
    LLAMA_METAL=0 \
    LLAMA_OPENBLAS=1
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Create data directory
RUN mkdir -p /app/data

# Environment defaults
ENV MODELS_DIR=/models

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app via entrypoint
CMD ["./docker-entrypoint.sh"]
