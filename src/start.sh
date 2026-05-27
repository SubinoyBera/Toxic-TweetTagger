#!/bin/bash
set -e

# Clear stale multiprocess metrics from previous runs (BEFORE app starts)
echo "Clearing stale Prometheus multiprocess metrics"
rm -rf /tmp/prometheus_metrics/*
mkdir -p /tmp/prometheus_metrics

# Substitute environment variables into prometheus config
envsubst < /app/src/prometheus.yaml > /tmp/prometheus_resolved.yaml

# Start Prometheus in background
echo "Starting Prometheus"
prometheus --config.file=/tmp/prometheus_resolved.yaml \
           --storage.tsdb.path=/tmp/prometheus_data &

PROMETHEUS_PID=$!
echo "Prometheus started with PID $PROMETHEUS_PID"

# Trap to kill Prometheus if the app exits
trap "echo 'Shutting down Prometheus...'; kill $PROMETHEUS_PID" EXIT

# Start FastAPI app (exec hands over control cleanly)
echo "Starting FastAPI app"
exec gunicorn \
    -k uvicorn.workers.UvicornWorker \
    src.app.main:app \
    --workers 2 \
    --bind 0.0.0.0:7860