"""rerank() implementation.

NOTE: as of this writing, the internal LiteLLM fork's rerank routing to the
self-hosted reranking backend may still be in progress. Response parsing
below assumes a Cohere-shaped payload (``results: [{index,
relevance_score}]``), matching LiteLLM's own upstream rerank convention —
this is an assumption, not a confirmed contract. Everything that depends on
the response shape is isolated in ``_parse_rerank_response`` so a real spec
mismatch is a one-function fix.
"""

from __future__ import annotations

import time

import httpx

from ._http import raise_for_status
from .types import RankedDocument, RerankResult, Timing, Usage

DEFAULT_RERANK_MODEL = "liviate/rerank"


async def rerank(http: httpx.AsyncClient, query: str, documents: list[str], model: str) -> RerankResult:
    start = time.perf_counter()
    response = await http.post(
        "/v1/rerank",
        json={"model": model, "query": query, "documents": documents},
    )
    await raise_for_status(response)
    elapsed_ms = (time.perf_counter() - start) * 1000

    ranked = _parse_rerank_response(response.json(), documents)

    return RerankResult(
        ranked=ranked,
        usage=Usage(rerank_docs=len(documents)),
        timing=Timing(rerank_ms=elapsed_ms),
    )


def _parse_rerank_response(payload: dict, documents: list[str]) -> list[RankedDocument]:
    """Assumes Cohere-shaped ``results: [{index, relevance_score}]``. VERIFY
    against the live endpoint before release — see module docstring."""
    results = payload["results"]
    ranked = [
        RankedDocument(
            text=documents[item["index"]],
            score=item["relevance_score"],
            index=item["index"],
        )
        for item in results
    ]
    ranked.sort(key=lambda d: d.score, reverse=True)
    return ranked
