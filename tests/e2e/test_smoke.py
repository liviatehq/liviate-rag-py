"""End-to-end smoke test against a real Liviate environment.

Skipped unless LIVIATE_E2E is set -- run manually before a release, not part
of the default CI suite. Requires LIVIATE_API_KEY and LIVIATE_E2E_MODEL
(an Inference model identifier the account has access to) in the
environment.
"""

from __future__ import annotations

import os

import pytest

from liviate_rag import RAGClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVIATE_E2E"),
    reason="set LIVIATE_E2E=1 (plus LIVIATE_API_KEY, LIVIATE_E2E_MODEL) to run the real-environment smoke test",
)


def test_ingest_then_query_smoke():
    client = RAGClient()
    collection = "sdk-smoke-test"

    ingest_result = client.ingest(
        source="Liviate offers a managed vector database, embedding, and reranking as separately billed products.",
        collection=collection,
        source_type="text",
    )
    assert ingest_result.chunks_created > 0

    result = client.query("What does Liviate offer?", collection=collection, model=os.environ["LIVIATE_E2E_MODEL"])
    assert result.answer
    assert result.sources
