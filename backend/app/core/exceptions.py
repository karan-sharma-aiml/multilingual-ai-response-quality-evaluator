"""Application-specific exception hierarchy."""

from typing import Any


class AppError(Exception):
    """Base class for all errors that map to a structured HTTP response."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidParameterError(AppError):
    status_code = 422
    error_code = "validation_error"


class AuthenticationError(AppError):
    status_code = 401
    error_code = "unauthorized"


class InvalidRequestError(AppError):
    status_code = 400
    error_code = "invalid_request"


class BatchFileError(InvalidRequestError):
    error_code = "invalid_batch_file"


class PayloadTooLargeError(AppError):
    status_code = 413
    error_code = "payload_too_large"


class ResultNotFoundError(AppError):
    status_code = 404
    error_code = "not_found"


class DatabaseOperationError(AppError):
    status_code = 500
    error_code = "database_error"


class GeminiError(AppError):
    """Base class for Gemini integration failures."""

    status_code = 502
    error_code = "gemini_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        details: Any | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message, details=details)
        if retryable is not None:
            self.retryable = retryable


class GeminiNotConfiguredError(GeminiError):
    status_code = 503
    error_code = "gemini_not_configured"


class GeminiTimeoutError(GeminiError):
    status_code = 504
    error_code = "gemini_timeout"
    retryable = True


class GeminiRateLimitError(GeminiError):
    status_code = 429
    error_code = "gemini_rate_limited"
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class GeminiAPIError(GeminiError):
    status_code = 502
    error_code = "gemini_api_error"


class GeminiInvalidResponseError(GeminiError):
    status_code = 502
    error_code = "gemini_invalid_response"
    retryable = True
