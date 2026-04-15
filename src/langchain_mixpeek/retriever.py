"""Mixpeek retrievers for LangChain."""

import asyncio
from functools import partial
from typing import List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from mixpeek import Mixpeek
from pydantic import Field, model_validator


class MixpeekRetriever(BaseRetriever):
    """LangChain retriever backed by a Mixpeek retriever pipeline.

    Searches across video, images, audio, and documents using
    Mixpeek's multi-stage retrieval pipelines.

    Example:
        .. code-block:: python

            from langchain_mixpeek import MixpeekRetriever

            retriever = MixpeekRetriever(
                api_key="mxp_...",
                retriever_id="ret_abc123",
                namespace="my-namespace",
                top_k=5,
            )
            docs = retriever.invoke("find the red cup")
    """

    api_key: str = Field(description="Mixpeek API key (mxp_...)")
    retriever_id: str = Field(description="Mixpeek retriever ID (ret_...)")
    namespace: str = Field(description="Mixpeek namespace to search")
    top_k: int = Field(default=10, description="Maximum number of results to return")
    content_field: str = Field(
        default="transcript_chunk",
        description="Metadata field to use as Document page_content",
    )
    filters: Optional[dict] = Field(
        default=None,
        description="Optional attribute filters to pass to the retriever",
    )

    def _build_client(self):
        return Mixpeek(api_key=self.api_key, namespace=self.namespace)

    def _execute(self, query: str) -> list:
        client = self._build_client()
        inputs = {"query": query}
        if self.filters:
            inputs["filters"] = self.filters
        response = client.retrievers.execute(
            retriever_id=self.retriever_id,
            inputs=inputs,
        )
        # The API returns {"results": [...], "status": ..., ...}
        if isinstance(response, dict):
            return response.get("results", [])
        return response

    def _extract_content(self, item: dict, metadata: dict) -> str:
        """Extract page_content from content_field, checking both metadata and top-level."""
        # Try metadata first (legacy format), then top-level (current API format)
        raw = metadata.pop(self.content_field, None) or item.get(self.content_field)
        if raw is None:
            return ""
        # If the field is a dict with a "text" key (e.g. trend_insight, brand_alignment),
        # extract the text value
        if isinstance(raw, dict) and "text" in raw:
            return str(raw["text"])
        return str(raw)

    def _results_to_documents(self, results: list) -> List[Document]:
        docs = []
        for item in results[: self.top_k]:
            metadata = dict(item.get("metadata") or {})
            # Merge top-level fields into metadata (current API puts fields at top level)
            for key in ("collection_id", "thumbnail_url", "_source_tier"):
                if key in item:
                    metadata[key] = item[key]
            metadata["document_id"] = item.get("document_id", "")
            metadata["score"] = item.get("score", 0.0)
            metadata["namespace"] = self.namespace
            content = self._extract_content(item, metadata)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        results = self._execute(query)
        return self._results_to_documents(results)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, partial(self._execute, query))
        return self._results_to_documents(results)


class AsyncMixpeekRetriever(MixpeekRetriever):
    """Async-first variant of MixpeekRetriever.

    Identical to :class:`MixpeekRetriever` but surfaces the async interface
    as the primary entry point. The sync fallback delegates to the event loop
    via ``run_in_executor`` so it is safe to call from either context.

    Example:
        .. code-block:: python

            from langchain_mixpeek import AsyncMixpeekRetriever

            retriever = AsyncMixpeekRetriever(
                api_key="mxp_...",
                retriever_id="ret_abc123",
                namespace="my-namespace",
            )
            docs = await retriever.ainvoke("find the red cup")
    """

    # All logic lives in MixpeekRetriever; this subclass exists as an explicit
    # async-oriented alias and to allow future divergence (e.g. true async HTTP).
