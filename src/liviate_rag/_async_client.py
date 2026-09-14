"""AsyncRAGClient — async-first implementation. RAGClient wraps this."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI

from ._config import resolve_api_key
from ._embed import DEFAULT_EMBED_MODEL
from ._embed import embed as _embed
from ._generate import generate as _generate
from ._generate import generate_stream as _generate_stream
from ._http import DEFAULT_BASE_URL, build_async_httpx_client, build_async_openai_client
from ._ingest.handlers import run_ingest, run_ingest_with_timeout
from ._pipeline import run_retrieval
from ._rerank import DEFAULT_RERANK_MODEL
from ._rerank import rerank as _rerank
from ._vectorstore import VectorStoreClient
from .types import AsyncIngestJob, EmbedResult, IngestResult, QueryResult, RerankResult, RetrieveResult


class AsyncRAGClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        exchange_url: str | None = None,
    ):
        """``exchange_url`` overrides where the vector store's token-exchange call goes (default:
        console.liviate.com -- see _vectorstore.py). It lives on a different host than
        ``base_url`` even in production, so pointing ``base_url`` at a non-default environment
        (e.g. a staging deployment) does *not* redirect it automatically; pass ``exchange_url``
        explicitly too in that case. Tests use this to point it at a local mock server."""
        self._api_key = resolve_api_key(api_key)
        self._base_url = base_url
        self._http: httpx.AsyncClient = build_async_httpx_client(self._api_key, base_url, timeout)
        self._openai: AsyncOpenAI = build_async_openai_client(self._api_key, base_url)
        vectorstore_kwargs = {}
        if exchange_url is not None:
            vectorstore_kwargs["exchange_url"] = exchange_url
        self._vectorstore = VectorStoreClient(self._api_key, timeout, **vectorstore_kwargs)

    async def close(self) -> None:
        await self._http.aclose()
        await self._openai.close()
        await self._vectorstore.close()

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
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> IngestResult | AsyncIngestJob:
        """Extracts text, chunks it, embeds each chunk, and upserts it into the vector store --
        entirely client-side (see _ingest/handlers.py). wait=False schedules this as a background
        task and returns immediately rather than leaving anything running server-side."""
        kwargs = dict(
            embed_client=self._openai, vectorstore=self._vectorstore, source=source, collection=collection,
            source_type=source_type, metadata=metadata, embed_model=embed_model,
        )
        if wait:
            return await run_ingest_with_timeout(**kwargs, timeout=timeout)

        job_id = str(uuid.uuid4())
        task = asyncio.create_task(run_ingest_with_timeout(**kwargs, timeout=timeout))

        async def _poll() -> IngestResult:
            return await task

        return AsyncIngestJob(job_ids=[job_id], collection=collection, _poll_fn=_poll)

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
        """Not yet implemented. Whole-site crawling (following internal links, respecting
        robots.txt, paging through up to max_pages) is a genuinely separate feature from
        ingest()'s single-source path -- it needs its own real engineering (crawl frontier,
        politeness/rate limiting, dedup), not a rushed version bolted onto ingest()'s pipeline.
        Raises rather than pretending to support this."""
        raise NotImplementedError(
            "ingest_site() is not yet implemented. Use ingest() with a list of individual page "
            "URLs in the meantime -- it already accepts a batch of sources in one call."
        )

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
            vectorstore=self._vectorstore, http=self._http, embed_client=self._openai, query=query,
            collection=collection, top_k=top_k, filter=filter, rerank_model=rerank_model,
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
