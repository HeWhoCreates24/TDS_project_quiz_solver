# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Playwright, PDF processing, and audio
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libwayland-client0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    libpango-1.0-0 \
    libcairo2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first for dependency caching
COPY pyproject.toml .

# Install uv package manager for faster dependency installation
RUN pip install --no-cache-dir uv

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Install Playwright browsers
RUN playwright install chromium --with-deps

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp/quiz_solver_cache

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health', timeout=5)" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
