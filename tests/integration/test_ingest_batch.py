"""Batch ingest() behavior: partial failure vs. total failure.

Per the API reference, one failed item in a batch must not abort the
others (aggregated into .warnings/.per_source[i].warnings). But if EVERY
item fails, nothing was ingested at all -- that's a total failure, not a
partial one, and must raise rather than return a success-shaped zero-chunk
result a caller could miss without inspecting .warnings.
"""

from __future__ import annotations

import pytest
from pytest_httpserver import HTTPServer

from liviate_rag.exceptions import LiviateError


async def test_batch_of_all_unclassifiable_items_raises(async_client):
    # Regression test: this previously returned successfully with
    # chunks_created=0 and the failures buried in .warnings -- confirmed live against
    # production before the fix.
    with pytest.raises(LiviateError, match="All 2 item"):
        await async_client.ingest(["not a path or url", "also not one"], collection="docs")


async def test_batch_partial_failure_still_warns_not_raises(async_client, mock_server: HTTPServer, tmp_path):
    # One good item (a real file, auto-detected) + one bad item (an unclassifiable raw string):
    # the good one should still succeed, with the bad one's failure surfaced as a warning --
    # NOT raised, since something did succeed.
    real_file = tmp_path / "notes.txt"
    real_file.write_text("some real text content")

    mock_server.expect_request("/v1/embeddings", method="POST").respond_with_json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "liviate/embedding",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
    )
    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_json(
        {
            "token": "scoped-jwt",
            "access": "rw",
            "expires_at": 9999999999,
            "collection_name": "testtenant__docs",
            "qdrant_url": mock_server.url_for("/").rstrip("/"),
        }
    )
    mock_server.expect_request("/collections/testtenant__docs/points", method="PUT").respond_with_json({})

    result = await async_client.ingest(["not a path or url", str(real_file)], collection="docs")

    assert result.chunks_created > 0
    assert len(result.warnings) == 1


async def test_empty_batch_does_not_raise(async_client):
    result = await async_client.ingest([], collection="docs")
    assert result.chunks_created == 0
    assert result.warnings == []
