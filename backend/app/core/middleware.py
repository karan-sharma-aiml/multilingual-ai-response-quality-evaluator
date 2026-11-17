"""HTTP middleware: request logging and response-time measurement."""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger("http")

REQUEST_ID_HEADER: str = "X-Request-ID"
PROCESS_TIME_HEADER: str = "X-Process-Time-Ms"
MS_PER_SECOND: float = 1000.0


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request and adds request-id / timing headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        logger.info("Request started: id=%s %s %s", request_id, request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            duration = (time.perf_counter() - started) * MS_PER_SECOND
            logger.exception(
                "Request crashed: id=%s %s %s duration_ms=%.1f",
                request_id, request.method, request.url.path, duration,
            )
            raise
        duration = (time.perf_counter() - started) * MS_PER_SECOND
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration:.2f}"
        logger.info(
            "Request finished: id=%s %s %s status=%d duration_ms=%.1f",
            request_id, request.method, request.url.path, response.status_code, duration,
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds defensive HTTP headers to every response."""

    def __init__(self, app, hsts: bool = False) -> None:  # noqa: ANN001
        super().__init__(app)
        self._hsts = hsts

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        if request.url.path not in DOC_PATHS:
            response.headers.setdefault("Cache-Control", "no-store")
        if self._hsts:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


DOC_PATHS: frozenset[str] = frozenset({"/docs", "/redoc", "/openapi.json"})
