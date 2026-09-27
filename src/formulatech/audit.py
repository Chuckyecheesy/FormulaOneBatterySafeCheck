"""Audit log: one JSON line per safety check (FR-7, spec/01-requirements.md).

Each record holds the inputs, calculated values, stage results, model version and a UTC
timestamp, so every decision can be audited later. Records are appended to
`logs/audit.jsonl`; set FORMULATECH_AUDIT_LOG to write them somewhere else.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any

from formulatech.config import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_LOG = REPO_ROOT / "logs" / "audit.jsonl"
_lock = threading.Lock()  # API requests run in a thread pool; keep each line whole


def audit_log_path() -> Path:
    return Path(os.environ.get("FORMULATECH_AUDIT_LOG", DEFAULT_AUDIT_LOG))


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with "Infinity", "-Infinity" or "NaN", which strict JSON parsers accept.

    A tiny charging duration makes dT/dt infinite; bare `Infinity` is not valid JSON.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_audit_record(record: dict[str, Any]) -> None:
    """Append one record as a strict-JSON line. A write failure is logged, never raised: it must not block the verdict."""
    line = json.dumps(_json_safe(record), default=str, ensure_ascii=False, allow_nan=False)
    path = audit_log_path()
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except OSError:
        logger.exception("Could not write the audit record to %s: %s", path, line)
