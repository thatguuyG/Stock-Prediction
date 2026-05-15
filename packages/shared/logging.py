"""Logging helpers — single JSON-ish line format, level driven by Settings."""
from __future__ import annotations

import logging
import sys

from packages.shared.config import get_settings

_configured = False


def configure_logging() -> None:
    global _configured  # pylint: disable=global-statement
    if _configured:
        return
    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
