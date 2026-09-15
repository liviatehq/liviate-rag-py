"""Result and data types returned by RAGClient / AsyncRAGClient.

Plain dataclasses rather than dicts, so IDEs autocomplete
``result.usage.tokens`` etc. instead of ``result["usage"]["tokens"]``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    """Billing/observability breakdown. Fields default to 0 so a result
    from a single step (e.g. standalone rerank()) doesn't need to populate
    fields that step didn't touch."""

    embed_tokens: int = 0
    rerank_docs: int = 0
    generation_tokens: int = 0


@dataclass
class Timing:
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0


@dataclass
class EmbedUsage:
    tokens: int


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    usage: EmbedUsage
    timing: Timing


@dataclass
class RankedDocument:
    text: str
    score: float
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    ranked: list[RankedDocument]
    usage: Usage
    timing: Timing


@dataclass
class RetrieveResult:
    sources: list[RankedDocument]
    usage: Usage
    timing: Timing


@dataclass
class QueryResult:
    answer: str
    sources: list[RankedDocument]
    usage: Usage
    timing: Timing


@dataclass
class IngestResult:
    chunks_created: int
    source_type: str
    collection: str
    warnings: list[str] = field(default_factory=list)
    per_source: list["IngestResult"] | None = None
    point_ids: list[str] = field(default_factory=list)
    """Vector-store point IDs this ingest wrote, so the content can later be
    removed with ``client.delete(collection, ids=result.point_ids)``."""
    """Populated only for batch ingest (one entry per input item)."""


@dataclass
class IngestJob:
    """Returned by RAGClient.ingest(..., wait=False) instead of blocking."""

    job_ids: list[str]
    collection: str
    _poll_fn: Callable[[], IngestResult]
    status: str = "queued"

    @property
    def job_id(self) -> str:
        return self.job_ids[0]

    def result(self) -> IngestResult:
        result = self._poll_fn()
        self.status = "done"
        return result

    def wait(self) -> None:
        self.result()


@dataclass
class AsyncIngestJob:
    """Returned by AsyncRAGClient.ingest(..., wait=False) instead of
    blocking. Same shape as IngestJob but result()/wait() are coroutines."""

    job_ids: list[str]
    collection: str
    _poll_fn: Callable[[], Awaitable[IngestResult]]
    status: str = "queued"

    @property
    def job_id(self) -> str:
        return self.job_ids[0]

    async def result(self) -> IngestResult:
        result = await self._poll_fn()
        self.status = "done"
        return result

    async def wait(self) -> None:
        await self.result()
