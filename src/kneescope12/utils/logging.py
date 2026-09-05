"""I configure a dedicated application logger for reproducible diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "kneescope12"


def configure_logging(
    level: str | int = "INFO", *, log_file: str | Path | None = None
) -> logging.Logger:
    """I configure and return the KneeScope-12 logger without touching the root logger."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """I return a child of the dedicated application logger."""

    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
