"""In-memory sliding-window rate limiting middleware.

State is per process. Behind several workers/instances use a shared store
(e.g. Redis) or a gateway-level limiter for exact global limits.
"""

import hashlib
import threading
import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import Settings
from app.core.error_handlers import error_payload
from app.core.logging import get_logger

logger = get_logger("rate_limit")

EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})
API_KEY_HEADER: str = "x-api-key"
FORWARDED_HEADER: str = "x-forwarded-for"
CLEANUP_INTERVAL_SECONDS: float = 60.0


class SlidingWindowLimiter:
    """Allows ``limit`` hits per ``window`` seconds for each key."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int, float]:
        """Register a hit; returns ``(allowed, remaining, seconds_until_reset)``."""
        current = time.monotonic() if now is None else now
        with self._lock:
            self._cleanup(current)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= current - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, 0, max(0.0, hits[0] + self.window - current)
            hits.append(current)
            reset = hits[0] + self.window - current
            return True, self.limit - len(hits), reset

    def _cleanup(self, now: float) -> None:
        if now - self._last_cleanup < CLEANUP_INTERVAL_SECONDS:
            return
        self._last_cleanup = now
        stale = [k for k, h in self._hits.items() if not h or h[-1] <= now - self.window]
        for key in stale:
            del self._hits[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rejects requests above the configured rate with HTTP 429."""

    def __init__(self, app, settings: Settings) -> None:  # noqa: ANN001
        super().__init__(app)
        self._enabled = settings.rate_limit_enabled
        self._trust_proxy = settings.trust_proxy_headers
        self._limiter = SlidingWindowLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get(API_KEY_HEADER)
        if api_key:
            return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        if self._trust_proxy:
            forwarded = request.headers.get(FORWARDED_HEADER)
            if forwarded:
                return "ip:" + forwarded.split(",")[0].strip()
        return "ip:" + (request.client.host if request.client else "unknown")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._enabled or request.url.path in EXEMPT_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        allowed, remaining, reset = self._limiter.check(self._client_key(request))
        headers = {
            "X-RateLimit-Limit": str(self._limiter.limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset) + 1),
        }
        if not allowed:
            logger.warning("Rate limit exceeded: %s %s", request.method, request.url.path)
            headers["Retry-After"] = str(int(reset) + 1)
            return JSONResponse(
                status_code=429,
                content=error_payload(
                    "rate_limit_exceeded",
                    "Too many requests. Please retry later.",
                    {"retry_after_seconds": int(reset) + 1},
                ),
                headers=headers,
            )
        response = await call_next(request)
        response.headers.update(headers)
        return response
