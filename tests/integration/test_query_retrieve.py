"""query() and retrieve() share _pipeline.run_retrieval() internally --
these tests assert that shared path stays intact (both produce identical
`sources`, and query() adds exactly one more call for generation).
"""

from __future__ import annotations

import json

import pytest
from pytest_httpserver import HTTPServer

from liviate_rag.exceptions import LiviateError


def _mock_retrieval_chain(mock_server: HTTPServer) -> None:
    mock_server.expect_request("/v1/embeddings", method="POST").respond_with_json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "liviate/embedding",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
    )
    # Real contract: the SDK first exchanges api_key for a short-lived, collection-scoped
    # credential (console's vectordb_exchange_token), then calls the vector store's OWN native
    # query API directly with that credential, at the collection's real (tenant-namespaced) name
    # and the data-plane URL the exchange response itself supplies -- see _vectorstore.py.
    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_json(
        {
            "token": "scoped-jwt-for-hotel-kirstine",
            "access": "r",
            "expires_at": 9999999999,
            "collection_name": "testtenant__hotel-kirstine",
            "qdrant_url": mock_server.url_for("/").rstrip("/"),
        }
    )
    mock_server.expect_request("/collections/testtenant__hotel-kirstine/points/query", method="POST").respond_with_json(
        {
            "result": {
                "points": [
                    {"id": "1", "score": 0.5, "payload": {"text": "We have free parking.", "metadata": {"page": 1}}},
                    {"id": "2", "score": 0.3, "payload": {"text": "Breakfast is included.", "metadata": {"page": 2}}},
                ]
            },
            "status": "ok",
            "time": 0.001,
        }
    )
    mock_server.expect_request("/v1/rerank", method="POST").respond_with_json(
        {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.4},
            ]
        }
    )


async def test_retrieve_returns_ranked_sources(async_client, mock_server: HTTPServer):
    _mock_retrieval_chain(mock_server)

    result = await async_client.retrieve("Har I parkering?", collection="hotel-kirstine", top_k=5)

    assert result.sources[0].text == "We have free parking."
    assert result.sources[0].metadata == {"page": 1}
    assert result.usage.embed_tokens == 3
    assert result.usage.rerank_docs == 2


async def test_query_reuses_retrieve_and_adds_generation(async_client, mock_server: HTTPServer):
    _mock_retrieval_chain(mock_server)
    mock_server.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Yes, free parking is available."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }
    )

    result = await async_client.query("Har I parkering?", collection="hotel-kirstine", model="test/model")

    assert result.answer == "Yes, free parking is available."
    assert result.sources[0].text == "We have free parking."
    assert result.usage.generation_tokens == 8
    assert result.timing.generate_ms >= 0


async def test_retrieve_passes_custom_embed_model_through(async_client, mock_server: HTTPServer):
    # Regression test: retrieve()/query() used to have no embed_model parameter at all, so a
    # collection ingested with a non-default embed model could never be queried correctly --
    # the query would always be embedded with the default model instead. Asserting on the
    # captured request body's "model" field (rather than an exact-body matcher, since the
    # openai client adds its own fields like encoding_format) means this only passes if
    # embed_model actually reaches the /v1/embeddings call.
    mock_server.expect_request("/v1/embeddings", method="POST").respond_with_json(
        {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "custom/embedding-v2",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
    )
    mock_server.expect_request("/api/tenancy/vectordb/exchange-token/", method="POST").respond_with_json(
        {
            "token": "scoped-jwt",
            "access": "r",
            "expires_at": 9999999999,
            "collection_name": "testtenant__hotel-kirstine",
            "qdrant_url": mock_server.url_for("/").rstrip("/"),
        }
    )
    mock_server.expect_request("/collections/testtenant__hotel-kirstine/points/query", method="POST").respond_with_json(
        {"result": {"points": []}}
    )

    result = await async_client.retrieve(
        "Har I parkering?", collection="hotel-kirstine", embed_model="custom/embedding-v2",
    )
    assert result.usage.embed_tokens == 3

    embed_requests = [req for req, _resp in mock_server.log if req.path == "/v1/embeddings"]
    assert len(embed_requests) == 1
    sent_body = json.loads(embed_requests[0].get_data())
    assert sent_body["model"] == "custom/embedding-v2"


async def test_retrieve_raises_on_point_with_no_text_payload(async_client, mock_server: HTTPServer):
    # A point written outside of ingest() (e.g. a direct upsert missing "text") must raise
    # clearly rather than silently returning an empty-text source -- see _pipeline.py.
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
            "access": "r",
            "expires_at": 9999999999,
            "collection_name": "testtenant__hotel-kirstine",
            "qdrant_url": mock_server.url_for("/").rstrip("/"),
        }
    )
    mock_server.expect_request("/collections/testtenant__hotel-kirstine/points/query", method="POST").respond_with_json(
        {"result": {"points": [{"id": "1", "score": 0.5, "payload": {"metadata": {"page": 1}}}]}}
    )

    with pytest.raises(LiviateError, match="no 'text' in its payload"):
        await async_client.retrieve("Har I parkering?", collection="hotel-kirstine", top_k=5)
