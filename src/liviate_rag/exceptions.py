"""Exception hierarchy for liviate-rag.

ingest() raises the builtin ``ValueError`` directly (not a subclass) when it
can't classify a source — matches the API reference literally. Every other
SDK-specific error inherits from ``LiviateError``.
"""

from __future__ import annotations


class LiviateError(Exception):
    """Base class for all liviate-rag errors (excluding the builtin
    ValueError raised by ingest() source classification — see module
    docstring)."""


class UnsupportedFileType(LiviateError):
    """A file's sniffed content type isn't one of the v1 supported types
    (.pdf, .docx, .md, .txt, .csv, .json, .html). OCR/image ingestion is
    out of scope for v1 and raises this rather than failing silently or
    half-parsing."""


class IngestTimeout(LiviateError):
    """ingest(..., wait=True) exceeded the timeout. Retry with wait=False
    and poll the returned IngestJob instead."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message
            or "ingest() timed out while waiting for completion. Retry with "
            "wait=False and poll the returned IngestJob instead."
        )


class PartialIngestError(LiviateError):
    """Some but not all chunks from a single source failed to ingest.

    NOT RAISED in v1: the project brief and the API reference both lean
    towards warning-carrying behavior (the readable parts still succeed),
    pending product confirmation — see IngestResult.warnings. This class is
    defined and exported so the surface exists once that's confirmed;
    nothing in this codebase currently raises it.
    """


class RateLimitError(LiviateError):
    """Raised on HTTP 429 responses from any Liviate backend."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after
