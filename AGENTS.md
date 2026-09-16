# AGENTS.md — liviate-rag-py

Context for coding agents (Claude Code, Copilot, Cursor, etc.) working in this repo.
This is a Python SDK for Liviate's RAG pipeline: ingest → embed → retrieve → rerank →
(optionally) generate. Read this before writing code against it.

## Setup

Current version: **0.1.4**. This is a young SDK (v0.1.x) — expect the API surface and the
gotchas below to shift between releases; check the installed version against this file if
something doesn't match.

```bash
pip install -e ".[dev]"    # installs from this checkout (repo work)
# or: pip install liviate-rag==0.1.4   # pin explicitly when installing from PyPI
pytest tests/unit tests/integration   # no network/credentials required
LIVIATE_E2E=1 pytest tests/e2e        # hits a real environment — run manually, not in CI loops
```

Auth: `RAGClient(api_key="...")` or set `LIVIATE_API_KEY` in the environment (get a key at
https://console.liviate.com). Don't hardcode keys.

## Core surface

```python
from liviate_rag import RAGClient, AsyncRAGClient

client = RAGClient()
client.ingest("handbook.pdf", collection="hotel-kirstine")
result = client.query("Har I parkering?", collection="hotel-kirstine", model="anthropic/claude-sonnet-5")
# result.answer / result.sources / result.usage / result.timing

context = client.retrieve("Har I parkering?", collection="hotel-kirstine", top_k=5)  # BYO-LLM path
```

`AsyncRAGClient` mirrors the same method surface for async codebases (use `async with`).

## Things that will bite you if you don't know them

- **`ingest()` detects source type from content, not filename.** A single call accepts a
  file, URL, raw text (`source_type="text"`), or file-like stream — never branch on file
  extension yourself. Supported: PDF, DOCX, MD, TXT, CSV, JSON, HTML. OCR/scanned images are
  explicitly unsupported for v1 (`UnsupportedFileType`, not a silent partial parse).
- **Whole-site ingestion:** pass a list of individual page URLs to `ingest()` — it already
  batches, so there's no separate site-crawl method (never was one shipped; the crawling
  itself is intentionally not this SDK's job — use whatever crawler you already have to
  produce the URL list).
- **Batch `ingest()` is partial-fail-tolerant.** If some items in a list fail but at least
  one succeeds, the call returns normally — check `result.warnings`. Only a *total* batch
  failure raises `LiviateError`.
- **Embedding model auto-detection is live, but young — set it explicitly until you've
  confirmed it for your own collections.** `retrieve()`/`query()` auto-resolve the
  `embed_model` a collection was ingested with when the backend has it on record (shipped
  server-side 2026-09-16). Collections created before that date won't have it recorded, and
  a mismatched embed model returns garbage or throws a dimension-mismatch error rather than
  a helpful message — so until you've verified auto-resolution for a given collection, keep
  passing `embed_model=` explicitly to `retrieve()`/`query()` matching what you used at
  `ingest()` time.
- **Two distinct exception families — don't collapse them:**
  - `liviate_rag.LiviateError` (and subclasses `APIError`, `RateLimitError`,
    `UnsupportedFileType`, `IngestTimeout`) = backend/runtime failures. `except LiviateError`
    catches all of these, including failures inside the `openai`-routed generation step.
  - Plain `ValueError` = caller mistakes (unclassifiable source, calling `delete()` with
    both or neither of `ids`/`filter`). These are bugs in the calling code — handle
    separately, don't swallow them under `except LiviateError`.
- **`delete()` takes exactly one of `ids` or `filter`, never both/neither:**
  ```python
  client.delete("hotel-kirstine", ids=result.point_ids)
  client.delete("hotel-kirstine", filter={"must": [...]})
  ```

## When building on this SDK

- Prefer `client.query()` when the SDK should also call the LLM; use `client.retrieve()`
  when the caller wants ranked context only and will call its own LLM.
- Wrap ingestion/query calls in `try/except LiviateError` for user-facing error handling;
  let `ValueError` surface during development as a signal of incorrect usage.
- Collection naming and `embed_model` choice should be decided once per project and reused
  consistently — mixing embed models across ingest/query calls into the same collection is
  the most common source of silent bugs right now (see above).
