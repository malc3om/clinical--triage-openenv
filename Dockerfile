# Stage 1: Build Dashboard
FROM node:20-slim AS builder
WORKDIR /app/frontend
COPY clinical-triage-dashboard/package*.json ./
RUN npm ci
COPY clinical-triage-dashboard/ ./
RUN npm run build

# Stage 2: Python server
FROM python:3.11-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built dashboard from Stage 1
COPY --from=builder /app/frontend/out /app/dashboard_out

# Install the package
RUN pip install -e .

# Ensure appuser owns the working directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Performance settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "clinical_triage_env.app:app", "--host", "0.0.0.0", "--port", "7860"]
