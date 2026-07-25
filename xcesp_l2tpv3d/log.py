"""Structured logging for xcesp-l2tpv3d.

Thin wrapper over the stdlib ``logging`` module so we can add richer
contextual formatting later (JSON, per-tunnel prefixes, structlog)
without touching every callsite.  systemd's journald consumes stderr
by default, so we log there.
"""

from __future__ import annotations

import logging
import sys


_LEVEL_MAP = {
    "debug":   logging.DEBUG,
    "info":    logging.INFO,
    "warning": logging.WARNING,
    "error":   logging.ERROR,
}


def configure(level: str = "info") -> None:
    """Set up root logging.

    ``level`` is one of debug|info|warning|error (case-insensitive).
    Called once at daemon startup after config load.
    """
    lvl = _LEVEL_MAP.get(level.lower(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
        force=True,   # override any prior basicConfig from tests / imports
    )


def get(name: str) -> logging.Logger:
    """Return a named logger under the ``xcesp-l2tpv3d`` root."""
    return logging.getLogger(f"xcesp-l2tpv3d.{name}")
