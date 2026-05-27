FROM python:3.11.11-slim-bookworm

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential wget gettext-base && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app/src

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r src/app/requirements.txt -r src/app/model/requirements.txt

# Prometheus multiprocess directory
RUN mkdir -p /tmp/prometheus_metrics && \
    chmod 777 /tmp/prometheus_metrics

# Install Prometheus
RUN wget https://github.com/prometheus/prometheus/releases/download/v2.51.0/prometheus-2.51.0.linux-amd64.tar.gz && \
    tar xvf prometheus-2.51.0.linux-amd64.tar.gz && \
    mv prometheus-2.51.0.linux-amd64/prometheus /usr/local/bin/ && \
    rm -rf prometheus-2.51.0.linux-amd64*

# Copy prometheus config (accessible to appuser)
COPY prometheus.yaml /app/prometheus.yaml

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Create non-root user and fix permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /tmp/prometheus_metrics /start.sh
USER appuser

EXPOSE 7860

ENV HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_metrics

# Only ENTRYPOINT
ENTRYPOINT ["/start.sh"]