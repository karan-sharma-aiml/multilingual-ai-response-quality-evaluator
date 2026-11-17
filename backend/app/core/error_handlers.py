"""Centralised exception handlers producing a uniform error payload."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger("errors")

VALIDATION_PREFIX: str = "Value error, "
LOCATION_ROOTS: frozenset[str] = frozenset({"body", "query", "path", "form", "header"})


def error_payload(code: str, message: str, details: object | None = None) -> dict[str, object]:
    """Build the standard error body."""
    return {"error": {"code": code, "message": message, "details": details}}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    level = logger.error if exc.status_code >= 500 else logger.warning
    level("App error: id=%s code=%s message=%s", _request_id(request), exc.error_code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc.error_code, exc.message, exc.details),
    )


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for item in exc.errors():
        location = [str(part) for part in item["loc"]]
        if location and location[0] in LOCATION_ROOTS:
            location = location[1:]
        details.append(
            {
                "field": ".".join(location) or "request",
                "message": str(item["msg"]).removeprefix(VALIDATION_PREFIX),
                "type": item["type"],
            }
        )
    logger.warning("Validation error: id=%s errors=%d", _request_id(request), len(details))
    return JSONResponse(
        status_code=422,
        content=error_payload("validation_error", "Request validation failed.", details),
    )


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning("HTTP error: id=%s status=%d detail=%s", _request_id(request), exc.status_code, exc.detail)
    code = "not_found" if exc.status_code == 404 else "http_error"
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(code, str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.error("Database error: id=%s %s", _request_id(request), exc.__class__.__name__)
    return JSONResponse(
        status_code=500,
        content=error_payload("database_error", "A database error occurred."),
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled error: id=%s %s: %s", _request_id(request), exc.__class__.__name__, exc,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content=error_payload("internal_error", "An unexpected error occurred."),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all handlers to ``app``."""
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(SQLAlchemyError, handle_database_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
