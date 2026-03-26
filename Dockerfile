# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps needed by sentence-transformers, torch, pdfminer, unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir wandb

# Copy source
COPY . .

# Pre-download the default embedding model so the first run is fast
# (comment out if you prefer lazy download or use a different model)
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF

# ChromaDB data lives in a volume; docs are mounted at runtime
VOLUME ["/app/chroma_db", "/app/docs"]

# Default command: show CLI help. Override in docker-compose or `docker run`.
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
