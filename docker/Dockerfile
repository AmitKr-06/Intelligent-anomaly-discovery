FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache

ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Install runtime dependencies
COPY docker/requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the application from the project-root build context
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser params.yaml ./params.yaml

# Runtime log directory
RUN mkdir -p /app/logs /tmp/numba_cache \
    && chown -R appuser:appuser /app/logs /tmp/numba_cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]