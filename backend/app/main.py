"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.dependencies import get_engine, shutdown_resources
from app.api.routes import analytics, batch, evaluation, health, reports, results
from app.core.config import Settings, get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from app.core.rate_limit import RateLimitMiddleware
from app.core.security import require_api_key
from app.database.init_db import initialize_database

APP_VERSION: str = "1.0.0"
GZIP_MIN_BYTES: int = 1024
ALLOWED_METHODS: list[str] = ["GET", "POST", "DELETE", "OPTIONS"]
ALLOWED_HEADERS: list[str] = ["Content-Type", "X-API-Key", "X-Request-ID"]
API_DESCRIPTION: str = (
    "Evaluates LLM responses in English, Hindi, French, Spanish and German using a "
    "rule-based engine, optionally combined with Gemini, and stores every evaluation."
)
OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "Health", "description": "Service status."},
    {"name": "Evaluation", "description": "Evaluate and compare responses."},
    {"name": "Batch", "description": "Evaluate a CSV file."},
    {"name": "Results", "description": "Browse and delete stored evaluations."},
    {"name": "Analytics", "description": "Aggregated quality statistics and trends."},
    {"name": "Reports", "description": "CSV, Excel and PDF exports."},
]


def create_app(settings: Settings | None = None, initialize_db: bool = True) -> FastAPI:
    """Build the application, register middleware, handlers and routers."""
    active_settings = settings or get_settings()
    setup_logging(active_settings)
    logger = get_logger("app")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if initialize_db and active_settings.auto_create_tables:
            initialize_database(get_engine())
        if not active_settings.api_key_list:
            logger.warning("API_KEYS is not set: API key authentication is DISABLED")
        logger.info(
            "%s v%s started (env=%s)", active_settings.app_name, APP_VERSION, active_settings.app_env
        )
        yield
        shutdown_resources()
        logger.info("Application stopped")

    app = FastAPI(
        title=active_settings.app_name,
        version=APP_VERSION,
        description=API_DESCRIPTION,
        debug=active_settings.debug,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs" if active_settings.enable_docs else None,
        redoc_url="/redoc" if active_settings.enable_docs else None,
        openapi_url="/openapi.json" if active_settings.enable_docs else None,
        lifespan=lifespan,
    )

    origins = active_settings.cors_origin_list
    # Middleware added last runs first (outermost): CORS -> gzip -> logging -> security -> rate limit.
    app.add_middleware(RateLimitMiddleware, settings=active_settings)
    app.add_middleware(SecurityHeadersMiddleware, hsts=active_settings.is_production)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=GZIP_MIN_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=ALLOWED_METHODS,
        allow_headers=ALLOWED_HEADERS,
        expose_headers=["X-Request-ID", "X-Process-Time-Ms", "X-RateLimit-Remaining", "Retry-After"],
    )
    register_exception_handlers(app)

    app.include_router(health.router)
    for router in (
        evaluation.router, batch.router, results.router, analytics.router, reports.router,
    ):
        app.include_router(router, dependencies=[Security(require_api_key)])
    return app


app = create_app()
