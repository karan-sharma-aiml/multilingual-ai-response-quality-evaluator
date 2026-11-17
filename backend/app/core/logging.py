"""Centralised logging configuration (console + rotating file)."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import Settings, get_settings
from app.utils.paths import PROJECT_ROOT

LOGGER_ROOT_NAME: str = "evaluator"
LOG_FILE_NAME: str = "evaluator.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE_MAX_BYTES: int = 5 * 1024 * 1024
LOG_FILE_BACKUP_COUNT: int = 3


def _resolve_log_dir(log_dir: str) -> Path:
    path = Path(log_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure the package logger with console and file handlers.

    The function is idempotent: existing handlers installed by a previous call
    are removed before new ones are attached.
    """
    active_settings = settings or get_settings()
    logger = logging.getLogger(LOGGER_ROOT_NAME)
    logger.setLevel(getattr(logging, active_settings.log_level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = _resolve_log_dir(active_settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / LOG_FILE_NAME,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, configuring logging on first use."""
    root = logging.getLogger(LOGGER_ROOT_NAME)
    if not root.handlers:
        setup_logging()
    return logging.getLogger(f"{LOGGER_ROOT_NAME}.{name}")
