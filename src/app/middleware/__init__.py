import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from src.core.logger import logging
from src.app.monitoring.http_metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_IN_PROGRESS


async def http_observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    method = request.method
    route = request.scope.get("route")
    path = route.path if route else request.url.path

    start_time = time.perf_counter()

    logging.info(f"[{request_id}] Incoming request {method} {path}")

    try:
        response = await call_next(request)
        status_code = response.status_code

    except Exception:
        duration = time.perf_counter() - start_time

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            path=path,
            status=500,
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            path=path,
        ).observe(duration)
                
        raise       # FastAPI will handle the error response

    duration = time.perf_counter() - start_time

    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        path=path,
        status=str(status_code),
    ).inc()

    HTTP_REQUESTS_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
    finally:
        HTTP_REQUESTS_IN_PROGRESS.dec()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method,
        path=path,
    ).observe(duration)

    logging.info("incoming_request",
                 extra={"request_id": request_id,
                        "method": method, 
                        "path": path
                    }
    )

    response.headers["X-Request-ID"] = request_id

    return response