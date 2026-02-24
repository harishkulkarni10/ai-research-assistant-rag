# Multi-stage Dockerfile for RAG ArXiv Research Assistant
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit for UI
RUN pip install --no-cache-dir streamlit==1.32.0

# Copy application code
COPY . .

# Expose ports
# 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "rag_arxiv_qa.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
