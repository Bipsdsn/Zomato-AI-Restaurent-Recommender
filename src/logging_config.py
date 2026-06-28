"""
logging_config.py -- Phase 11: Centralized application logging.

Provides a single place to configure console logging for the app. The level
is driven by the LOG_LEVEL environment variable (default INFO). Pipeline
modules use `logging.getLogger(__name__)` so their output is controllable
instead of relying on bare print() calls.

Note: this is separate from `logger.py`, which writes structured JSONL request
logs for observability/evaluation. This module handles human-readable console
diagnostics.
"""

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """
    Configure the root console logger once. Safe to call multiple times.

    Args:
        level: Optional explicit level name (e.g. "DEBUG"). Falls back to the
               LOG_LEVEL env var, then "INFO".
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(log_level)
    # Avoid duplicate handlers if something already configured logging.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configuration is handled by configure_logging)."""
    return logging.getLogger(name)
