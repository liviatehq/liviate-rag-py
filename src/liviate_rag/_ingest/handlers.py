"""Ingest source I/O: HTTP submission + job polling for ingest()/ingest_site().

ASSUMPTION (not yet verified against a real endpoint spec — flagged per the
plan, same treatment as the rerank and vector-search assumptions): every
ingest submission goes through POST /v1/ingest (or /v1/ingest/site for
whole-site crawls) and returns {"job_id": ..., "status": ...}; job status is
polled via GET /v1/ingest/jobs/{job_id} returning {"status":
"queued"|"running"|"done"|"failed", "result": {...} | null, "error": str |
null}. wait=True just means "poll until done/failed before returning" —
there is no separate synchronous ingest code path.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx
import magic

from .._http import raise_for_status
from ..exceptions import IngestTimeout, LiviateError, UnsupportedFileType
from ..types import IngestResult
from .detect import Classification, classify

_DEFAULT_INGEST_TIMEOUT_S = 120.0
_POLL_INTERVAL_S = 1.0

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


async def _submit_json(http: httpx.AsyncClient, payload: dict) -> str:
    response = await http.post("/v1/ingest", json=payload)
    await raise_for_status(response)
    return response.json()["job_id"]


async def _submit_file_bytes(
    http: httpx.AsyncClient, data: bytes, collection: str, metadata: dict | None, filename: str,
) -> str:
    source_type = sniff_file_type(data)
    response = await http.post(
        "/v1/ingest",
        data={"collection": collection, "metadata": json.dumps(metadata or {}), "source_type": source_type},
        files={"file": (filename, data)},
    )
    await raise_for_status(response)
    return response.json()["job_id"]


async def _submit_url(http: httpx.AsyncClient, url: str, collection: str, metadata: dict | None) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as web:
        head = await web.head(url, timeout=15.0)
        content_type = head.headers.get("content-type", "").split(";")[0].strip().lower()

    if content_type in ("", "text/html"):
        return await _submit_json(http, {"url": url, "collection": collection, "metadata": metadata or {}})

    async with httpx.AsyncClient(follow_redirects=True) as web:
        download = await web.get(url, timeout=60.0)
        download.raise_for_status()
    filename = url.rsplit("/", 1)[-1] or "download"
    return await _submit_file_bytes(http, download.content, collection, metadata, filename)


async def _submit_one(
    http: httpx.AsyncClient, classification: Classification, collection: str, metadata: dict | None,
) -> str:
    kind = classification.kind
    if kind == "text":
        return await _submit_json(http, {"text": classification.value, "collection": collection, "metadata": metadata or {}})
    if kind == "file":
        path = classification.value
        return await _submit_file_bytes(http, path.read_bytes(), collection, metadata, filename=path.name)
    if kind == "stream":
        data = classification.value.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return await _submit_file_bytes(http, data, collection, metadata, filename="stream")
    if kind == "url":
        return await _submit_url(http, classification.value, collection, metadata)
    raise AssertionError(f"unreachable classification kind: {kind}")


async def _fetch_job_status(http: httpx.AsyncClient, job_id: str) -> dict:
    response = await http.get(f"/v1/ingest/jobs/{job_id}")
    await raise_for_status(response)
    return response.json()


async def wait_for_jobs(http: httpx.AsyncClient, job_ids: list[str], collection: str, timeout: float) -> IngestResult:
    if not job_ids:
        return IngestResult(chunks_created=0, source_type="batch", collection=collection, warnings=[], per_source=[])

    deadline = time.monotonic() + timeout
    payloads: dict[str, dict] = {}
    pending = set(job_ids)
    while pending:
        if time.monotonic() > deadline:
            raise IngestTimeout()
        for job_id in list(pending):
            payload = await _fetch_job_status(http, job_id)
            if payload["status"] in ("done", "failed"):
                payloads[job_id] = payload
                pending.discard(job_id)
        if pending:
            await asyncio.sleep(_POLL_INTERVAL_S)

    return _aggregate(payloads, job_ids, collection)


def _aggregate(payloads: dict[str, dict], job_ids: list[str], collection: str) -> IngestResult:
    is_batch = len(job_ids) > 1
    per_source: list[IngestResult] = []
    warnings: list[str] = []
    total_chunks = 0

    for job_id in job_ids:
        payload = payloads[job_id]
        if payload["status"] == "failed":
            error = payload.get("error") or "ingest job failed"
            if not is_batch:
                raise LiviateError(error)
            item = IngestResult(chunks_created=0, source_type=payload.get("source_type", "unknown"), collection=collection, warnings=[error])
        else:
            r = payload["result"]
            item = IngestResult(
                chunks_created=r["chunks_created"], source_type=r["source_type"],
                collection=collection, warnings=r.get("warnings", []),
            )
            total_chunks += item.chunks_created
        per_source.append(item)
        warnings.extend(item.warnings)

    if not is_batch:
        return per_source[0]

    return IngestResult(chunks_created=total_chunks, source_type="batch", collection=collection, warnings=warnings, per_source=per_source)


async def ingest_source(
    *, http: httpx.AsyncClient, source: Any, collection: str, source_type: str,
    wait: bool, metadata: dict | None, timeout: float | None,
) -> IngestResult | list[str]:
    """Returns an IngestResult when wait=True, or the submitted job_ids when
    wait=False (the caller wraps those into an IngestJob/AsyncIngestJob).

    Batch semantics: one failed item must not abort the others (per the API
    reference) — submission failures (bad path, unsupported file type, ...)
    are caught per item and folded into the aggregated result's warnings /
    per_source rather than raised, as long as it's a multi-item batch. A
    single (non-batch) source still raises directly.
    """
    classification = classify(source, source_type)
    is_batch = classification.kind == "batch"

    if is_batch:
        job_ids: list[str] = []
        local_errors: list[str] = []
        for item in classification.value:
            try:
                item_classification = classify(item, source_type)
                job_ids.append(await _submit_one(http, item_classification, collection, metadata))
            except (ValueError, UnsupportedFileType, httpx.HTTPError) as exc:
                local_errors.append(str(exc))
    else:
        job_ids = [await _submit_one(http, classification, collection, metadata)]
        local_errors = []

    if not wait:
        return job_ids

    effective_timeout = timeout if timeout is not None else _DEFAULT_INGEST_TIMEOUT_S
    result = await wait_for_jobs(http, job_ids, collection, effective_timeout)

    if not local_errors:
        return result

    error_items = [
        IngestResult(chunks_created=0, source_type="unknown", collection=collection, warnings=[e])
        for e in local_errors
    ]
    return IngestResult(
        chunks_created=result.chunks_created,
        source_type="batch",
        collection=collection,
        warnings=result.warnings + local_errors,
        per_source=(result.per_source or [result]) + error_items,
    )
