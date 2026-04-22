from __future__ import annotations

"""Lightweight JSON-lines logging helpers (Pinecone backend).

This is duplicated from the Chroma backend so each backend can run standalone.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def setup_logger(log_dir: str | Path) -> logging.Logger:
    """Create (or return) an app logger that writes JSON lines to `app.log`."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aicsrg")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(Path(log_dir) / "app.log", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def log_event(logger: logging.Logger, event: str, payload: Mapping[str, Any]) -> None:
    """Log a structured event as a single JSON object line."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    logger.info(json.dumps(record, ensure_ascii=False))
