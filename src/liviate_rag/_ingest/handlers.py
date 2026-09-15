"""Ingest pipeline: extract -> chunk -> embed -> upsert.

Fully client-side and synchronous -- there is no server-side ingest job.
A source (file/text/url/stream) is turned into plain text (extract.py),
split into chunks (chunk.py), embedded via the same embed() the rest of
the package already uses, then written directly to the vector store via
VectorStoreClient.upsert() (the same token-exchange path vector_search()
uses). wait=False still returns immediately -- the work is scheduled as
a background asyncio task rather than left running server-side, and the
returned job object awaits that task on result()/wait().
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any

import httpx
import magic
from openai import AsyncOpenAI

from .._embed import embed as _embed
from ..exceptions import IngestTimeout, LiviateError, UnsupportedFileType
from ..types import IngestResult
from .chunk import chunk_text
from .detect import Classification, classify
from .extract import extract_text

_DEFAULT_INGEST_TIMEOUT_S = 120.0

_MIME_MAP = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/html": "html",
    "application/json": "json",
    "text/csv": "csv",
}


def sniff_file_type(data: bytes) -> str:
    """Detects file type from actual bytes (python-magic), never a filename
    extension — per the brief. libmagic can't reliably distinguish
    markdown/plain/csv/json once outside the well-known MIME types above,
    so those fall back to light content sniffing rather than guessing from
    a name."""
    mime = magic.from_buffer(data, mime=True)
    if mime in _MIME_MAP:
        return _MIME_MAP[mime]

    if mime.startswith("text/"):
        text = data.decode("utf-8", errors="ignore").strip()
        if text.startswith("{") or text.startswith("["):
            try:
                json.loads(text)
                return "json"
            except json.JSONDecodeError:
                pass
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].count(",") > 0 and lines[0].count(",") == lines[1].count(","):
            return "csv"
        if re.search(r"^#{1,6}\s", text, re.MULTILINE) or "```" in text:
            return "md"
        return "txt"

    raise UnsupportedFileType(
        f"Unsupported file type (detected content type: {mime!r}). Supported "
        "types: pdf, docx, md, txt, csv, json, html. Image/OCR ingestion is "
        "not supported."
    )


async def _text_and_type_for(classification: Classification) -> tuple[str, str]:
    kind = classification.kind
    if kind == "text":
        return classification.value, "text"
    if kind == "file":
        data = classification.value.read_bytes()
        source_type = sniff_file_type(data)
        return extract_text(data, source_type), source_type
    if kind == "stream":
        data = classification.value.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        source_type = sniff_file_type(data)
        return extract_text(data, source_type), source_type
    if kind == "url":
        return await _fetch_url_text(classification.value)
    raise AssertionError(f"unreachable classification kind: {kind}")


async def _fetch_url_text(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(follow_redirects=True) as web:
        response = await web.get(url, timeout=60.0)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type in ("", "text/html"):
        return extract_text(response.content, "html"), "html"
    source_type = sniff_file_type(response.content)
    return extract_text(response.content, source_type), source_type


async def _ingest_one(
    embed_client: AsyncOpenAI,
    vectorstore: Any,
    classification: Classification,
    collection: str,
    metadata: dict | None,
    embed_model: str,
) -> IngestResult:
    text, source_type = await _text_and_type_for(classification)
    chunks = chunk_text(text)
    if not chunks:
        return IngestResult(
            chunks_created=0, source_type=source_type, collection=collection,
            warnings=["no extractable text content"],
        )

    embed_result = await _embed(embed_client, chunks, embed_model)
    point_ids = [str(uuid.uuid4()) for _ in chunks]
    points = [
        {"id": point_id, "vector": vector, "payload": {"text": chunk, "metadata": metadata or {}}}
        for point_id, chunk, vector in zip(point_ids, chunks, embed_result.vectors)
    ]
    await vectorstore.upsert(collection, points, embed_model=embed_model)
    return IngestResult(
        chunks_created=len(chunks), source_type=source_type, collection=collection,
        warnings=[], point_ids=point_ids,
    )


async def _ingest_batch(
    embed_client: AsyncOpenAI,
    vectorstore: Any,
    items: list[Any],
    collection: str,
    metadata: dict | None,
    embed_model: str,
    source_type: str,
) -> IngestResult:
    per_source: list[IngestResult] = []
    warnings: list[str] = []
    total_chunks = 0
    for item in items:
        try:
            item_classification = classify(item, source_type)
            result = await _ingest_one(embed_client, vectorstore, item_classification, collection, metadata, embed_model)
        except (ValueError, LiviateError, httpx.HTTPError) as exc:
            result = IngestResult(chunks_created=0, source_type="unknown", collection=collection, warnings=[str(exc)])
        per_source.append(result)
        warnings.extend(result.warnings)
        total_chunks += result.chunks_created
    point_ids = [pid for r in per_source for pid in r.point_ids]

    # If every item in a non-empty batch failed, nothing was ingested -- that's a total
    # failure, not the "one bad item shouldn't abort the others" partial-failure case the
    # per-item warning behavior above exists for. Raise instead of returning a success-shaped
    # zero-chunk result a caller could easily miss without inspecting .warnings.
    if items and total_chunks == 0 and warnings:
        raise LiviateError(
            f"All {len(items)} item(s) in this batch failed to ingest -- nothing was written. "
            f"First error: {warnings[0]}"
        )

    return IngestResult(
        chunks_created=total_chunks, source_type="batch", collection=collection,
        warnings=warnings, per_source=per_source, point_ids=point_ids,
    )


async def run_ingest(
    *,
    embed_client: AsyncOpenAI,
    vectorstore: Any,
    source: Any,
    collection: str,
    source_type: str,
    metadata: dict | None,
    embed_model: str,
) -> IngestResult:
    """The actual synchronous ingest work -- extract, chunk, embed, upsert. Shared by both the
    wait=True (awaited directly) and wait=False (scheduled as a background task) paths."""
    classification = classify(source, source_type)
    if classification.kind == "batch":
        return await _ingest_batch(embed_client, vectorstore, classification.value, collection, metadata, embed_model, source_type)
    return await _ingest_one(embed_client, vectorstore, classification, collection, metadata, embed_model)


async def run_ingest_with_timeout(
    *,
    embed_client: AsyncOpenAI,
    vectorstore: Any,
    source: Any,
    collection: str,
    source_type: str,
    metadata: dict | None,
    embed_model: str,
    timeout: float | None,
) -> IngestResult:
    effective_timeout = timeout if timeout is not None else _DEFAULT_INGEST_TIMEOUT_S
    try:
        return await asyncio.wait_for(
            run_ingest(
                embed_client=embed_client, vectorstore=vectorstore, source=source, collection=collection,
                source_type=source_type, metadata=metadata, embed_model=embed_model,
            ),
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError as exc:
        raise IngestTimeout() from exc
