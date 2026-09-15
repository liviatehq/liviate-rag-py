"""Unit tests for the pure ingest() source-classification dispatch logic.

No network or real file I/O beyond tmp_path fixtures — see
liviate_rag/_ingest/detect.py's module docstring for why this logic is kept
pure and tested in isolation.
"""

from __future__ import annotations

import io

import pytest

from liviate_rag._ingest.detect import classify


def test_batch_list_detected():
    result = classify(["a.pdf", "b.pdf"])
    assert result.kind == "batch"
    assert result.value == ["a.pdf", "b.pdf"]


def test_batch_detected_regardless_of_explicit_source_type():
    # Regression test: batch detection used to only run inside the "auto" branch, so
    # classify(["text a", "text b"], source_type="text") raised ValueError("requires source to
    # be a str, got list") instead of recognizing a two-item batch.
    for source_type in ("text", "file", "url"):
        result = classify(["a", "b"], source_type=source_type)
        assert result.kind == "batch"
        assert result.value == ["a", "b"]


def test_batch_tuple_detected():
    result = classify(("a.pdf", "b.pdf"))
    assert result.kind == "batch"


def test_existing_local_path_detected_as_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello")
    result = classify(str(file_path))
    assert result.kind == "file"
    assert result.value == file_path


def test_url_string_detected():
    result = classify("https://example.com/page")
    assert result.kind == "url"
    assert result.value == "https://example.com/page"

    result_http = classify("http://example.com/page")
    assert result_http.kind == "url"


def test_file_like_object_detected_as_stream():
    stream = io.BytesIO(b"content")
    result = classify(stream)
    assert result.kind == "stream"
    assert result.value is stream


def test_ambiguous_nonexistent_string_raises_value_error():
    with pytest.raises(ValueError):
        classify("this/path/does/not/exist.txt")


def test_relative_nonexistent_path_raises_value_error():
    with pytest.raises(ValueError):
        classify("relative/typo/path.md")


def test_raw_text_without_explicit_source_type_is_not_silently_accepted():
    with pytest.raises(ValueError):
        classify("just some raw text content, not a path or url")


def test_raw_text_with_explicit_source_type_accepted():
    result = classify("just some raw text", source_type="text")
    assert result.kind == "text"
    assert result.value == "just some raw text"


def test_source_type_text_rejects_non_string():
    with pytest.raises(ValueError):
        classify(12345, source_type="text")


def test_unclassifiable_object_raises_value_error():
    with pytest.raises(ValueError):
        classify(object())


def test_explicit_source_type_file_requires_existing_path(tmp_path):
    with pytest.raises(ValueError):
        classify(str(tmp_path / "missing.txt"), source_type="file")


def test_explicit_source_type_url_requires_http_scheme():
    with pytest.raises(ValueError):
        classify("not-a-url", source_type="url")


def test_unknown_source_type_raises_value_error():
    with pytest.raises(ValueError):
        classify("anything", source_type="bogus")
