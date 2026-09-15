"""Exception hierarchy for liviate-rag.

Two distinct kinds of error, on purpose:

- **Caller mistakes** (bad arguments, a source ingest() can't classify)
  raise the builtin ``ValueError`` directly, never a ``LiviateError``
  subclass. Examples: ``ingest()`` given a source it can't classify,
  ``delete()`` given both/neither of ``ids``/``filter``. These are bugs in
  the calling code, not something a backend outage could ever cause, so
  ``except LiviateError`` deliberately does NOT catch them -- catch
  ``ValueError`` separately (or fix the call) if you need to handle these.
- **Backend/runtime failures** (a bad response from the Liviate API, a
  timeout, an unsupported file's content) raise ``LiviateError`` or one of
  its subclasses below. ``except LiviateError`` reliably catches every one
  of these, regardless of which HTTP transport made the call (the plain
  httpx-based calls and the ones routed through the ``openai`` client for
  embed()/generation both get wrapped the same way -- see
  ``_http.raise_for_status`` and ``_http.translate_openai_errors``).
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
    """Raised on HTTP 429 responses from any Liviate backend, including
    ones surfaced through the openai client (embed(), query()'s generation
    step) -- wrapped into this so `except RateLimitError` (or the broader
    `except LiviateError`) catches it regardless of which transport hit
    the limit."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class APIError(LiviateError):
    """Any other non-2xx response from a Liviate backend (not a rate limit
    -- see RateLimitError). Wraps both the plain httpx-based calls (ingest,
    vector store, rerank) and errors surfaced through the openai client
    (embed(), generation), so `except LiviateError` reliably catches
    backend failures no matter which transport made the call."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
