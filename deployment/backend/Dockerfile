FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    INDEX_DIR=/app/artifacts/faiss_index

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY deployment/backend/requirements.txt /app/deployment/backend/requirements.txt
# CPU wheels avoid downloading an unused CUDA runtime for this small POC.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r deployment/backend/requirements.txt
COPY rag_core.py retrieval.py /app/
COPY deployment/backend/app.py /app/deployment/backend/app.py
COPY artifacts/faiss_index /app/artifacts/faiss_index
# Cache the exact checkpoint pinned by retrieval.py and verify the saved index.
RUN python -c "from retrieval import load_index; load_index('artifacts/faiss_index')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app /opt/hf_cache
USER appuser
EXPOSE 7860
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:7860", "deployment.backend.app:create_app()"]
