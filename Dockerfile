FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for compilation & libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies from backend/
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK and spaCy model baseline data
RUN python -m nltk.downloader punkt stopwords wordnet
RUN python -m spacy download en_core_web_sm

# Copy backend application source code
COPY backend/ .

# Expose port (Render automatically injects $PORT at runtime, defaults to 8000)
ENV PORT=8000
EXPOSE 8000

# Run uvicorn bound to 0.0.0.0 and dynamically read $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
