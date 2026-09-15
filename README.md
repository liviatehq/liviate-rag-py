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

Whole-site crawling (`ingest_site()`) is **not yet implemented** — it
raises `NotImplementedError`. Use `ingest()` with a list of individual page
URLs in the meantime; it already accepts a batch of sources in one call.

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit tests/integration   # no network/credentials required
LIVIATE_E2E=1 pytest tests/e2e         # real environment, run manually
```
