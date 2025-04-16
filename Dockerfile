# Multi-stage build to minimize image size
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pysqlite3-binary

### Final stage with minimal image
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create .streamlit directory with proper permissions
RUN mkdir -p /app/.streamlit && \
    chmod 777 /app/.streamlit

# Fix the torch.classes.__path__ issue
RUN echo 'torch.classes.__path__ = []' > /app/.streamlit/config.py && \
    chmod 644 /app/.streamlit/config.py

# Copy application code (using wildcard to avoid missing file errors)
COPY *.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create data directory structure
RUN mkdir -p /app/data/chroma_db_llamaparse-openai

# Copy only the OpenAI vector store (the default one used)
COPY data/chroma_db_llamaparse-openai/ /app/data/chroma_db_llamaparse-openai/

# Copy PDF files from data directory
COPY data/*.pdf /app/data/

# Create images directory and copy only the required logo
RUN mkdir -p /app/images
COPY images/LOGO_UPBEAT.jpg images/estonia.jpg images/finland.png /app/images/
# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]