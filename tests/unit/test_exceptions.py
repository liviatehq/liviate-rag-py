"""Sanity checks on the exception hierarchy shape."""

from __future__ import annotations

from liviate_rag.exceptions import (
    IngestTimeout,
    LiviateError,
    PartialIngestError,
    RateLimitError,
    UnsupportedFileType,
)


def test_all_custom_exceptions_inherit_liviate_error():
    for exc_cls in (UnsupportedFileType, IngestTimeout, PartialIngestError, RateLimitError):
        assert issubclass(exc_cls, LiviateError)


def test_liviate_error_is_not_a_value_error():
    # ingest()'s classification failures raise the builtin ValueError
    # directly, not a LiviateError subclass -- confirm the hierarchies stay
    # separate so a broad `except LiviateError` doesn't silently swallow
    # classification bugs.
    assert not issubclass(LiviateError, ValueError)


def test_rate_limit_error_carries_retry_after():
    err = RateLimitError("rate limited", retry_after=12.5)
    assert err.retry_after == 12.5


def test_ingest_timeout_default_message_mentions_wait_false():
    err = IngestTimeout()
    assert "wait=False" in str(err)
