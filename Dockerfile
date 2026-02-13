# Face-Based Attendance System - Backend Dockerfile
# Multi-stage build for optimized production image

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt


# ==============================================================================
# Stage 2: Production
# ==============================================================================
FROM python:3.10-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY backend/ ./backend/
COPY models/ ./models/

# Create data directories
RUN mkdir -p /app/data/enrollments /app/data/attendance

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models/exported/mobilefacenet.onnx
ENV EMBEDDINGS_DIR=/app/data/enrollments
ENV ATTENDANCE_LOG_DIR=/app/data/attendance

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# ==============================================================================
# Stage 3: Development (optional)
# ==============================================================================
FROM production as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    isort \
    flake8

# Enable hot reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
