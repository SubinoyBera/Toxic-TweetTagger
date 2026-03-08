import time
import uuid
from fastapi import Request
from src.core.logger import logging
from src.app.monitoring.http_metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_IN_PROGRESS

async def http_observability_middleware(request: Request, call_next):
    # Skip Prometheus metrics endpoint
    if request.url.path == "/api/metrics":
        return await call_next(request)

    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    method = request.method
    route = request.scope.get("route")
    path = route.path if route else request.url.path

    start_time = time.perf_counter()

    logging.info(f"[{request_id}] Incoming request {method} {path}")

    HTTP_REQUESTS_IN_PROGRESS.inc()

    try:
        response = await call_next(request)
        status_code = response.status_code

    except Exception:
        duration = time.perf_counter() - start_time

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            path=path,
            status="500",
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            path=path,
        ).observe(duration)

        HTTP_REQUESTS_IN_PROGRESS.dec()
        raise

    duration = time.perf_counter() - start_time

    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        path=path,
        status=str(status_code),
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method,
        path=path,
    ).observe(duration)

    HTTP_REQUESTS_IN_PROGRESS.dec()

    response.headers["X-Request-ID"] = request_id

    return response