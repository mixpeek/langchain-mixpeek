"""Mixpeek toolkit — exposes multiple agent tools from a single store."""

import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_mixpeek.vectorstore import MixpeekVectorStore


class _MixpeekBaseTool(BaseTool):
    """Base class for toolkit tools — holds a reference to the store."""

    store: MixpeekVectorStore = Field(exclude=True)
    return_direct: bool = False


class MixpeekSearchTool(_MixpeekBaseTool):
    """Search multimodal content (video, images, audio, documents)."""

    name: str = "mixpeek_search"
    description: str = (
        "Search across video, image, audio, and document content. "
        "Input is a natural language query. Returns ranked results with "
        "document IDs, scores, and content."
    )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            docs = self.store.similarity_search(query)
            results = []
            for doc in docs:
                results.append({
                    "document_id": doc.metadata.get("document_id", ""),
                    "score": round(doc.metadata.get("score", 0.0), 4),
                    "content": doc.page_content,
                    "collection_id": doc.metadata.get("collection_id", ""),
                })
            return json.dumps(results, indent=2)
        except Exception as exc:
            return f"Search failed: {exc}"


class MixpeekIngestTool(_MixpeekBaseTool):
    """Ingest content into Mixpeek for processing."""

    name: str = "mixpeek_ingest"
    description: str = (
        "Upload content to Mixpeek for indexing and feature extraction. "
        "Input should be a JSON object with 'type' ('text', 'image', 'video', "
        "'audio', 'pdf', 'excel') and either 'data' (for text) or 'url' "
        "(for files). Optional 'metadata' dict. "
        'Example: {"type": "text", "data": "hello world", "metadata": {"source": "chat"}}'
    )

    def _run(self, input_str: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
            blob_type = params.get("type", "text")
            metadata = params.get("metadata")
            meta_list = [metadata] if metadata else None

            if blob_type == "text":
                data = params.get("data", "")
                ids = self.store.add_texts([data], metadatas=meta_list)
            else:
                url = params.get("url", "")
                ids = self.store.add_urls([url], metadatas=meta_list, blob_type=blob_type)

            return json.dumps({"object_ids": ids, "status": "uploaded"})
        except Exception as exc:
            return f"Ingest failed: {exc}"


class MixpeekClassifyTool(_MixpeekBaseTool):
    """Run taxonomy classification on documents."""

    name: str = "mixpeek_classify"
    description: str = (
        "Classify documents using a Mixpeek taxonomy. Input should be a JSON "
        "object with 'taxonomy_id' (the taxonomy to execute). "
        "Returns classification results. "
        'Example: {"taxonomy_id": "tax_abc123"}'
    )

    def _run(self, input_str: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
            taxonomy_id = params.get("taxonomy_id", "")
            result = self.store.execute_taxonomy(taxonomy_id)
            return json.dumps(result, default=str, indent=2)
        except Exception as exc:
            return f"Classification failed: {exc}"


class MixpeekClusterTool(_MixpeekBaseTool):
    """Group similar documents into clusters."""

    name: str = "mixpeek_cluster"
    description: str = (
        "Create and execute document clustering. Input should be a JSON "
        "object with optional 'cluster_type' ('vector' or 'attribute'), "
        "'algorithm' (e.g. 'kmeans'), and 'n_clusters'. "
        "If 'cluster_id' is provided, fetches existing cluster groups instead. "
        'Example: {"cluster_type": "vector", "algorithm": "kmeans", "n_clusters": 5}'
    )

    def _run(self, input_str: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str

            # If cluster_id provided, get existing groups
            if "cluster_id" in params:
                result = self.store.get_cluster_groups(params["cluster_id"])
                return json.dumps(result, default=str, indent=2)

            # Otherwise create + execute
            algorithm = params.get("algorithm", "kmeans")
            n_clusters = params.get("n_clusters", 5)
            cluster = self.store.create_cluster(
                cluster_type=params.get("cluster_type", "vector"),
                vector_config={
                    "algorithm": algorithm,
                    "algorithm_params": {"n_clusters": n_clusters},
                },
            )
            cluster_id = cluster.get("cluster_id", "")
            if cluster_id:
                self.store.execute_cluster(cluster_id)
            return json.dumps({"cluster_id": cluster_id, "status": "executing"})
        except Exception as exc:
            return f"Clustering failed: {exc}"


class MixpeekAlertTool(_MixpeekBaseTool):
    """Set up monitoring alerts for document matches."""

    name: str = "mixpeek_alert"
    description: str = (
        "Create an alert that monitors for matching documents and sends "
        "notifications. Input should be a JSON object with 'name' and "
        "optional 'webhook_url'. If 'alert_id' is provided, fetches results "
        "for an existing alert instead. "
        'Example: {"name": "brand-watch", "webhook_url": "https://example.com/hook"}'
    )

    def _run(self, input_str: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str

            # If alert_id provided, get results
            if "alert_id" in params:
                result = self.store.get_alert_results(params["alert_id"])
                return json.dumps(result, default=str, indent=2)

            # Otherwise create
            name = params.get("name", "agent-alert")
            notification_config: Dict[str, Any] = {
                "include_matches": True,
                "include_scores": True,
                "channels": [],
            }
            if "webhook_url" in params:
                notification_config["channels"].append({
                    "channel_type": "webhook",
                    "config": {"url": params["webhook_url"]},
                })
            if "slack_channel" in params:
                notification_config["channels"].append({
                    "channel_type": "slack",
                    "channel_id": params["slack_channel"],
                })
            if "email" in params:
                notification_config["channels"].append({
                    "channel_type": "email",
                    "config": {"to": params["email"]},
                })

            result = self.store.create_alert(
                name=name,
                notification_config=notification_config,
            )
            return json.dumps(result, default=str, indent=2)
        except Exception as exc:
            return f"Alert creation failed: {exc}"


class MixpeekProcessTool(_MixpeekBaseTool):
    """Trigger collection processing after ingesting content."""

    name: str = "mixpeek_process"
    description: str = (
        "Trigger the collection to process newly uploaded content. "
        "Call this after using mixpeek_ingest to start feature extraction "
        "(embedding, OCR, transcription, face detection, etc.). "
        "No input required — just call with an empty string."
    )

    def _run(self, query: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            result = self.store.trigger_processing()
            return json.dumps(result, default=str, indent=2)
        except Exception as exc:
            return f"Processing trigger failed: {exc}"


class MixpeekToolkit:
    """Multi-tool toolkit that gives agents full Mixpeek capabilities.

    Exposes search, ingest, classify, cluster, alert, and process tools
    from a single configuration. Use with LangGraph's ``create_react_agent``
    or any LangChain agent framework.

    Example:
        .. code-block:: python

            from langchain_mixpeek import MixpeekToolkit
            from langgraph.prebuilt import create_react_agent

            toolkit = MixpeekToolkit(
                api_key="mxp_...",
                namespace="my-namespace",
                bucket_id="bkt_abc123",
                collection_id="col_def456",
                retriever_id="ret_ghi789",
            )

            agent = create_react_agent(llm, toolkit.get_tools())
            agent.invoke({
                "messages": [("user", "Scan these URLs and alert me about counterfeits")]
            })
    """

    def __init__(self, store: Optional[MixpeekVectorStore] = None, **kwargs: Any):
        """Create a toolkit.

        Args:
            store: An existing MixpeekVectorStore instance. If not provided,
                one is created from ``kwargs`` (api_key, namespace, bucket_id,
                collection_id, retriever_id).
        """
        if store is not None:
            self._store = store
        else:
            self._store = MixpeekVectorStore(**kwargs)

    @property
    def store(self) -> MixpeekVectorStore:
        return self._store

    def get_tools(
        self,
        actions: Optional[List[str]] = None,
    ) -> List[BaseTool]:
        """Return the list of tools for an agent.

        Args:
            actions: Optional subset of tool names to include. Defaults to
                all tools. Valid names: ``"search"``, ``"ingest"``,
                ``"process"``, ``"classify"``, ``"cluster"``, ``"alert"``.

        Returns:
            List of LangChain tools.
        """
        all_tools: Dict[str, BaseTool] = {
            "search": MixpeekSearchTool(store=self._store),
            "ingest": MixpeekIngestTool(store=self._store),
            "process": MixpeekProcessTool(store=self._store),
            "classify": MixpeekClassifyTool(store=self._store),
            "cluster": MixpeekClusterTool(store=self._store),
            "alert": MixpeekAlertTool(store=self._store),
        }

        if actions is None:
            return list(all_tools.values())

        return [all_tools[name] for name in actions if name in all_tools]
