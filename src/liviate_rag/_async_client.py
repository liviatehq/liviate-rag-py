"""AsyncRAGClient — async-first implementation. RAGClient wraps this."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI

from ._config import resolve_api_key
from ._embed import DEFAULT_EMBED_MODEL
from ._embed import embed as _embed
from ._generate import generate as _generate
from ._generate import generate_stream as _generate_stream
from ._http import DEFAULT_BASE_URL, build_async_httpx_client, build_async_openai_client, raise_for_status
from ._ingest.handlers import _DEFAULT_INGEST_TIMEOUT_S, ingest_source, wait_for_jobs
from ._pipeline import run_retrieval
from ._rerank import DEFAULT_RERANK_MODEL
from ._rerank import rerank as _rerank
from .types import AsyncIngestJob, EmbedResult, IngestResult, QueryResult, RerankResult, RetrieveResult


class AsyncRAGClient:
    def __init__(self, api_key: str | None = None, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0):
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url
        self._http: httpx.AsyncClient = build_async_httpx_client(self._api_key, base_url, timeout)
        self._openai: AsyncOpenAI = build_async_openai_client(self._api_key, base_url)

    async def close(self) -> None:
        await self._http.aclose()
        await self._openai.close()

    async def __aenter__(self) -> "AsyncRAGClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- ingest ------------------------------------------------------

    async def ingest(
        self,
        source,
        collection: str,
        *,
        source_type: str = "auto",
        wait: bool = True,
        metadata: dict | None = None,
        timeout: float | None = None,
    ) -> IngestResult | AsyncIngestJob:
        outcome = await ingest_source(
            http=self._http, source=source, collection=collection,
            source_type=source_type, wait=wait, metadata=metadata, timeout=timeout,
        )
        if wait:
            return outcome  # type: ignore[return-value]

        job_ids = outcome  # type: ignore[assignment]
        effective_timeout = timeout if timeout is not None else _DEFAULT_INGEST_TIMEOUT_S

        async def _poll() -> IngestResult:
            return await wait_for_jobs(self._http, job_ids, collection, effective_timeout)

        return AsyncIngestJob(job_ids=job_ids, collection=collection, _poll_fn=_poll)

    async def ingest_site(
        self,
        source: str,
        collection: str,
        *,
        max_pages: int = 50,
        wait: bool = False,
        metadata: dict | None = None,
        timeout: float | None = None,
    ) -> IngestResult | AsyncIngestJob:
        """Whole-site crawl, kept separate from ingest() by design (see
        brief) — following internal links and respecting robots.txt is
        assumed to happen server-side (POST /v1/ingest/site), not in this
        SDK. Defaults to wait=False since crawls are long-running and a
        blocking default would be a surprising foot-gun — the opposite
        tradeoff from ingest()'s single-page default.
        """
        response = await self._http.post(
            "/v1/ingest/site",
            json={"url": source, "collection": collection, "max_pages": max_pages, "metadata": metadata or {}},
        )
        await raise_for_status(response)
        job_id = response.json()["job_id"]
        effective_timeout = timeout if timeout is not None else _DEFAULT_INGEST_TIMEOUT_S

        async def _poll() -> IngestResult:
            return await wait_for_jobs(self._http, [job_id], collection, effective_timeout)

        if not wait:
            return AsyncIngestJob(job_ids=[job_id], collection=collection, _poll_fn=_poll)
        return await _poll()

    # -- embed / rerank ------------------------------------------------

    async def embed(self, texts: list[str], model: str = DEFAULT_EMBED_MODEL) -> EmbedResult:
        return await _embed(self._openai, texts, model)

    async def rerank(self, query: str, documents: list[str], model: str = DEFAULT_RERANK_MODEL) -> RerankResult:
        return await _rerank(self._http, query, documents, model)

    # -- retrieve / query ------------------------------------------------

    async def retrieve(
        self,
        query: str,
        collection: str,
        *,
        top_k: int = 5,
        filter: dict | None = None,
        rerank_model: str | None = DEFAULT_RERANK_MODEL,
    ) -> RetrieveResult:
        return await run_retrieval(
            http=self._http, embed_client=self._openai, query=query, collection=collection,
            top_k=top_k, filter=filter, rerank_model=rerank_model,
        )

    async def query(
        self,
        query: str,
        collection: str,
        *,
        model: str,
        top_k: int = 5,
        filter: dict | None = None,
        stream: bool = False,
    ) -> QueryResult | AsyncIterator[str]:
        """``model`` has no Liviate default: generation is deliberately kept
        out of the bundled RAG product (see brief) so the caller always
        names their own Inference model explicitly."""
        retrieval = await self.retrieve(query, collection, top_k=top_k, filter=filter)

        if stream:
            return _generate_stream(self._openai, query, retrieval.sources, model)

        answer, gen_usage, gen_timing = await _generate(self._openai, query, retrieval.sources, model)
        usage = retrieval.usage
        usage.generation_tokens = gen_usage.generation_tokens
        timing = retrieval.timing
        timing.generate_ms = gen_timing.generate_ms
        return QueryResult(answer=answer, sources=retrieval.sources, usage=usage, timing=timing)
