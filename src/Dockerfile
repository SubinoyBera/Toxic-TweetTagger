FROM python:3.11.11-slim-bookworm

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/src

# install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r src/app/requirements.txt -r src/app/model/requirements.txt

# prometheus multiprocess directory
RUN mkdir -p /tmp/prometheus_metrics && \
    chmod 777 /tmp/prometheus_metrics

# startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# create non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 7860

ENV HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_metrics

ENTRYPOINT ["/start.sh"]

CMD ["gunicorn", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "src.app.main:app", \
     "--workers", "2", \
     "--bind", "0.0.0.0:7860"]