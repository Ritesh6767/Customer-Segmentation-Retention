

# Official slim Python base (reproducible & minimal)
FROM python:3.11-slim

# Metadata
LABEL maintainer="your.email@example.com"
LABEL description="Customer Segmentation ML Pipeline with Streamlit Dashboard"
LABEL version="1.0.0"

# System deps (matplotlib needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 mluser

WORKDIR /app

# Copy & install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY --chown=mluser:mluser . .

# Create runtime directories
RUN mkdir -p outputs models_saved data \
    && chown -R mluser:mluser /app

USER mluser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Entrypoint – runs training then launches Streamlit
ENTRYPOINT ["sh", "-c"]
CMD ["python train.py && streamlit run streamlit_app/app.py \
      --server.port=8501 \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false"]
