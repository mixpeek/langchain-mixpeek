"""Mixpeek tool for LangChain agents."""

import json
from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from mixpeek import Mixpeek
from pydantic import Field


class MixpeekTool(BaseTool):
    """LangChain tool for searching multimodal content via Mixpeek.

    Use this when the agent needs to search video, images, audio, or document
    content. Returns a JSON string of ranked results with source, timestamps,
    and relevance scores.

    Example:
        .. code-block:: python

            from langchain_mixpeek import MixpeekTool

            tool = MixpeekTool(
                api_key="mxp_...",
                retriever_id="ret_abc123",
                namespace="my-namespace",
                top_k=5,
            )
            result = tool.invoke("find the red cup")
    """

    name: str = "mixpeek_search"
    description: str = (
        "Search across video, image, audio, and document content. "
        "Input should be a natural language query describing what to find. "
        "Returns ranked results with source, timestamps, and relevance scores."
    )
    api_key: str = Field(description="Mixpeek API key (mxp_...)")
    retriever_id: str = Field(description="Mixpeek retriever ID (ret_...)")
    namespace: str = Field(description="Mixpeek namespace to search")
    top_k: int = Field(default=5, description="Maximum number of results to return")
    content_field: str = Field(
        default="text",
        description="Metadata field to surface as 'content' in each result",
    )
    return_direct: bool = False

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        client = Mixpeek(api_key=self.api_key, namespace=self.namespace)
        try:
            response = client.retrievers.execute(
                retriever_id=self.retriever_id,
                inputs={"query": query},
            )
            # The API returns {"results": [...], "status": ..., ...}
            if isinstance(response, dict):
                results = response.get("results", [])
            else:
                results = response
            trimmed = []
            for item in results[: self.top_k]:
                meta = dict(item.get("metadata") or {})
                # Extract content from metadata (legacy) or top-level (current API)
                raw_content = meta.get(self.content_field) or item.get(self.content_field)
                if isinstance(raw_content, dict) and "text" in raw_content:
                    content = raw_content["text"]
                else:
                    content = raw_content or meta.get("text", "")
                trimmed.append(
                    {
                        "document_id": item.get("document_id"),
                        "score": round(item.get("score", 0.0), 4),
                        "source": meta.get("source_url") or meta.get("file_name", ""),
                        "timestamp": meta.get("timestamp", ""),
                        "content": content,
                    }
                )
            return json.dumps(trimmed, indent=2)
        except Exception as exc:  # noqa: BLE001
            return f"Search failed: {exc}"
