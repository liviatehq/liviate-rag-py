"""Pure, I/O-free source classification for ingest().

No network calls, no file reads beyond an existence check. This isolation
is what makes the dispatch logic unit-testable without mocking the
filesystem or network — see tests/unit/test_ingest_detect.py.

Detection order (source_type="auto"):
1. list/tuple -> batch
2. existing local path -> file
3. http(s):// string -> url
4. object with .read() -> stream
5. anything else -> ValueError (raw strings are never silently treated as
   text; source_type="text" must be explicit)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Kind = Literal["batch", "file", "url", "stream", "text"]


@dataclass
class Classification:
    kind: Kind
    value: Any


def classify(source: Any, source_type: str = "auto") -> Classification:
    if source_type == "text":
        if not isinstance(source, str):
            raise ValueError(
                f"source_type='text' requires source to be a str, got {type(source).__name__}."
            )
        return Classification("text", source)

    if source_type == "file":
        return Classification("file", _require_existing_path(source))

    if source_type == "url":
        return Classification("url", _require_url_string(source))

    if source_type != "auto":
        raise ValueError(f"Unknown source_type: {source_type!r}")

    if isinstance(source, (list, tuple)):
        return Classification("batch", list(source))

    if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
        return Classification("file", Path(source))

    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        return Classification("url", source)

    if hasattr(source, "read") and callable(getattr(source, "read", None)):
        return Classification("stream", source)

    if isinstance(source, str):
        raise ValueError(
            "Could not classify string source as an existing file path or a "
            "URL. If this is raw text content, pass source_type='text' "
            "explicitly — raw strings are never treated as text content by "
            "default."
        )

    raise ValueError(
        f"Could not classify source of type {type(source).__name__}. Pass "
        "source_type explicitly ('file', 'url', or 'text')."
    )


def _require_existing_path(source: Any) -> Path:
    path = Path(source)
    if not path.exists():
        raise ValueError(f"source_type='file' but path does not exist: {source!r}")
    return path


def _require_url_string(source: Any) -> str:
    if not (isinstance(source, str) and (source.startswith("http://") or source.startswith("https://"))):
        raise ValueError(f"source_type='url' requires an http(s):// string, got {source!r}")
    return source
