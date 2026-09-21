"""Shared logging setup used by agents, API, and the CLI."""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def setup_logging(level: str | int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
