FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/root/.cache/torch

# UNCOMMENTED: build-essential is required to compile basicsr
# libgl1 and libglib2.0-0 are required to prevent OpenCV runtime crashes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Upgrade pip
# 2. IMPORTANT: Pin setuptools to < 70.0.0 so basicsr can successfully build
RUN pip install --no-cache-dir --upgrade pip wheel "setuptools<70.0.0"

# REMOVED --no-build-isolation to allow standard PEP 517 dependency resolution
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories
RUN mkdir -p storage/uploads storage/outputs weights

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]