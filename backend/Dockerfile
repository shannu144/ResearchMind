FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid pulling 2GB+ NVIDIA CUDA packages.
# This dramatically reduces import time from ~30s to ~3s on Render free tier.
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining python dependencies (torch already satisfied above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK and spaCy model baseline data
RUN python -m nltk.downloader punkt stopwords wordnet
RUN python -m spacy download en_core_web_sm

# Copy backend application source code
COPY . .

# Expose port
ENV PORT=8000
EXPOSE 8000

# Run uvicorn with dynamic $PORT from Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
