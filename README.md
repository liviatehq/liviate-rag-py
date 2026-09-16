# liviate-rag

Client SDK for Liviate's RAG stack — ingest, embed, retrieve, rerank, and
(optionally) generate — in a few lines of Python.

```bash
pip install liviate-rag
```

## Quickstart

```python
from liviate_rag import RAGClient

client = RAGClient(api_key="...")  # or set LIVIATE_API_KEY

client.ingest("handbook.pdf", collection="hotel-kirstine")

result = client.query("Har I parkering?", collection="hotel-kirstine", model="anthropic/claude-sonnet-5")
print(result.answer)
print(result.sources)
print(result.usage)
print(result.timing)
```

Want ranked context only, and to call your own LLM?

```python
context = client.retrieve("Har I parkering?", collection="hotel-kirstine", top_k=5)
```

An `AsyncRAGClient` with the same method surface is available for async
codebases:

```python
from liviate_rag import AsyncRAGClient

async with AsyncRAGClient() as client:
    result = await client.query("...", collection="...", model="...")
```

## Ingest

`ingest()` handles a single file, URL, text string, or file-like stream —
detecting which one automatically from content, never a filename
extension. See `_ingest/detect.py` for the exact detection order.

### Supported file types

| Type | Extension |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| Markdown | `.md` |
| Plain text | `.txt` |
| CSV | `.csv` |
| JSON | `.json` |
| HTML | `.html` |

Plus raw text (`source_type="text"`) and URLs (a single page is scraped or
downloaded automatically depending on its content type).

OCR / scanned images are explicitly out of scope for v1 — `ingest()`
raises `UnsupportedFileType` rather than failing silently or half-parsing.

Whole-site crawling isn't part of this SDK — that's a genuinely different
problem (robots.txt compliance, politeness/rate limiting, avoiding crawl
traps) than ingesting sources you already have. Crawl with whatever tool
you already use, then pass the resulting URL list to `ingest()` — it
already accepts a batch of sources in one call.

If you ingest into a collection with a non-default `embed_model=`,
`retrieve()`/`query()` try to resolve the right model automatically —
today's backend doesn't yet record which model created a collection, so
until it does, pass the same `embed_model=` explicitly when querying it. A
query embedded with a different model than the collection's vectors either
returns garbage or fails outright on a dimension mismatch, so this is
worth getting right rather than relying on the not-yet-live auto-detection.

## Removing content

```python
result = client.ingest("handbook.pdf", collection="hotel-kirstine")
client.delete("hotel-kirstine", ids=result.point_ids)          # by id
client.delete("hotel-kirstine", filter={"must": [...]})        # by metadata filter
```

## Errors

Two distinct kinds of error, on purpose:

- **Backend/runtime failures** — a bad response from the Liviate API, a
  timeout, an unsupported file type — always surface as
  `liviate_rag.LiviateError` or a subclass (`APIError`, `RateLimitError`,
  `UnsupportedFileType`, `IngestTimeout`), whatever the underlying
  transport, including calls routed through the `openai` client for
  `embed()`/`query()`'s generation step. `except LiviateError` reliably
  catches all of these.
- **Caller mistakes** — a source `ingest()` can't classify, calling
  `delete()` with both/neither of `ids`/`filter` — raise the builtin
  `ValueError` directly, never a `LiviateError`. These are bugs in the
  calling code, not something `except LiviateError` is meant to catch;
  handle `ValueError` separately if you need to.

`ingest()` on a batch (list of sources) is worth calling out specifically:
if some items fail but at least one succeeds, the call still returns
normally — failures show up in `result.warnings`, per the API reference's
"one bad item shouldn't abort the batch" design. But if *every* item in
the batch fails, nothing was ingested, and that raises `LiviateError`
rather than returning a quiet zero-chunk "success."

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit tests/integration   # no network/credentials required
LIVIATE_E2E=1 pytest tests/e2e         # real environment, run manually
```
