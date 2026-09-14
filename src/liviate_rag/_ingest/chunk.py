"""Text chunking for ingest().

A simple, dependency-free splitter: fills chunks up to ``chunk_size``
characters, preferring to break on a paragraph boundary, then a sentence
boundary, then a plain word boundary as a fallback, and carries
``overlap`` characters of context into the next chunk so a fact split
across a boundary isn't lost. Good enough for the sizes real documents
chunk into (typically a few hundred to a couple thousand characters) --
not attempting token-exact sizing, which would require pulling in a
tokenizer this package otherwise has no use for.
"""

from __future__ import annotations

import re

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _best_break(text, start, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _best_break(text: str, start: int, end: int) -> int:
    window = text[start:end]

    last_para = None
    for m in _PARAGRAPH_BREAK.finditer(window):
        last_para = m
    if last_para is not None and last_para.end() > len(window) * 0.5:
        return start + last_para.end()

    last_sentence = None
    for m in _SENTENCE_BREAK.finditer(window):
        last_sentence = m
    if last_sentence is not None and last_sentence.end() > len(window) * 0.5:
        return start + last_sentence.end()

    last_space = window.rfind(" ")
    if last_space > len(window) * 0.5:
        return start + last_space + 1

    return end
