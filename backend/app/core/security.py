"""API key authentication."""

import secrets

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError

API_KEY_HEADER_NAME: str = "X-API-Key"

api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description="API key. Authentication is enforced when API_KEYS is configured.",
)


def is_valid_api_key(candidate: str | None, valid_keys: list[str]) -> bool:
    """Constant-time comparison of ``candidate`` against every accepted key."""
    if not candidate:
        return False
    candidate_bytes = candidate.encode("utf-8")
    matches = [secrets.compare_digest(candidate_bytes, key.encode("utf-8")) for key in valid_keys]
    return any(matches)


def require_api_key(
    api_key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency; a no-op when no API keys are configured."""
    keys = settings.api_key_list
    if keys and not is_valid_api_key(api_key, keys):
        raise AuthenticationError("Invalid or missing API key.")
