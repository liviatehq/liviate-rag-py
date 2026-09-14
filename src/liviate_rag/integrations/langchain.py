"""LangChain adapter: wraps RAGClient.retrieve() as a LangChain retriever.

Optional integration — langchain-core is not a hard dependency of the base
package, only imported when this module is imported directly (install with
``pip install liviate-rag[langchain]``).
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:
    raise ImportError(
        "liviate_rag.integrations.langchain requires langchain-core. "
        "Install it with: pip install liviate-rag[langchain]"
    ) from exc

from .._client import RAGClient


class LiviateRetriever(BaseRetriever):
    client: RAGClient
    collection: str
    top_k: int = 5

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        result = self.client.retrieve(query, self.collection, top_k=self.top_k)
        return [
            Document(page_content=doc.text, metadata={**doc.metadata, "score": doc.score})
            for doc in result.sources
        ]
