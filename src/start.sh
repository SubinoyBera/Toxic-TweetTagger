#!/bin/sh

# Clear Prometheus multiprocess metrics
rm -rf /tmp/prometheus_metrics/*

# Start the application
exec "$@"