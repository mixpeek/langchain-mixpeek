"""Mixpeek vector store for LangChain — add and search multimodal documents."""

import json
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from mixpeek import Mixpeek
from pydantic import Field


class MixpeekVectorStore(VectorStore):
    """LangChain VectorStore backed by Mixpeek.

    Supports adding documents (via bucket upload + collection trigger) and
    searching via Mixpeek's multi-stage retrieval pipelines.

    Example:
        .. code-block:: python

            from langchain_mixpeek import MixpeekVectorStore

            store = MixpeekVectorStore(
                api_key="mxp_...",
                namespace="my-namespace",
                bucket_id="bkt_abc123",
                collection_id="col_def456",
                retriever_id="ret_ghi789",
            )

            # Add documents
            store.add_texts(["hello world"], metadatas=[{"source": "test"}])

            # Search
            docs = store.similarity_search("hello")
    """

    api_key: str
    namespace: str
    bucket_id: str
    collection_id: str
    retriever_id: str
    content_field: str = "text"

    blob_property: str = "content"

    def __init__(
        self,
        api_key: str,
        namespace: str,
        bucket_id: str,
        collection_id: str,
        retriever_id: str,
        content_field: str = "text",
        blob_property: str = "content",
        **kwargs: Any,
    ):
        self.api_key = api_key
        self.namespace = namespace
        self.bucket_id = bucket_id
        self.collection_id = collection_id
        self.retriever_id = retriever_id
        self.content_field = content_field
        self.blob_property = blob_property
        self._client = Mixpeek(api_key=api_key, namespace=namespace)
        self._base_url = "https://api.mixpeek.com/v1"

    @property
    def embeddings(self):
        """Mixpeek handles embeddings server-side via feature extractors."""
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload text content to a Mixpeek bucket.

        Each text is uploaded as an object. The collection's feature extractor
        processes it asynchronously (embedding, OCR, etc.).

        Args:
            texts: Text content to upload.
            metadatas: Optional metadata dicts for each text.
            **kwargs: Passed to bucket upload (e.g., ``url`` for file URLs).

        Returns:
            List of object IDs from the upload.
        """
        object_ids = []
        meta_list = metadatas or [{}] * len(list(texts))
        for text, meta in zip(texts, meta_list):
            result = self._upload_object(
                blobs=[{"property": self.blob_property, "type": "text", "data": text}],
                metadata=meta,
            )
            object_ids.append(result.get("object_id", ""))
        return object_ids

    def add_urls(
        self,
        urls: List[str],
        metadatas: Optional[List[dict]] = None,
        blob_type: str = "image",
        **kwargs: Any,
    ) -> List[str]:
        """Upload files by URL to a Mixpeek bucket.

        Use this for images, videos, PDFs, and other files accessible via URL.
        For type-specific convenience methods, see :meth:`add_images`,
        :meth:`add_videos`, :meth:`add_audio`, :meth:`add_pdfs`, and
        :meth:`add_excel`.

        Args:
            urls: File URLs to ingest.
            metadatas: Optional metadata dicts for each URL.
            blob_type: Blob type — one of ``"image"``, ``"video"``,
                ``"audio"``, ``"text"``, ``"pdf"``, ``"excel"``.

        Returns:
            List of object IDs from the upload.
        """
        object_ids = []
        meta_list = metadatas or [{}] * len(urls)
        for url, meta in zip(urls, meta_list):
            result = self._upload_object(
                blobs=[{"property": self.blob_property, "type": blob_type, "url": url}],
                metadata=meta,
            )
            object_ids.append(result.get("object_id", ""))
        return object_ids

    def add_images(
        self,
        urls: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload images by URL.

        Supported formats: JPG, PNG, GIF, WebP, etc.
        Processed by image, multimodal, face, or IP-safety extractors.
        """
        return self.add_urls(urls, metadatas=metadatas, blob_type="image", **kwargs)

    def add_videos(
        self,
        urls: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload videos by URL.

        Supported formats: MP4, MOV, AVI, WebM, etc.
        Processed by multimodal extractor (scene detection, transcription,
        frame extraction, OCR, thumbnails).
        """
        return self.add_urls(urls, metadatas=metadatas, blob_type="video", **kwargs)

    def add_audio(
        self,
        urls: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload audio files by URL.

        Supported formats: MP3, WAV, FLAC, OGG, etc.
        Processed by audio fingerprint extractor (CLAP embeddings).
        """
        return self.add_urls(urls, metadatas=metadatas, blob_type="audio", **kwargs)

    def add_pdfs(
        self,
        urls: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload PDF documents by URL.

        Processed by document graph extractor (OCR, layout analysis,
        block classification, text extraction with bounding boxes).
        """
        return self.add_urls(urls, metadatas=metadatas, blob_type="pdf", **kwargs)

    def add_excel(
        self,
        urls: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Upload spreadsheets by URL.

        Supported formats: XLSX, XLS, CSV.
        """
        return self.add_urls(urls, metadatas=metadatas, blob_type="excel", **kwargs)

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _api_request(
        self,
        path: str,
        method: str = "POST",
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to the Mixpeek API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Namespace": self.namespace,
        }
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            method=method,
            headers=headers,
            data=data,
        )
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read().decode())

    def _upload_object(
        self,
        blobs: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Upload an object to the bucket using the blobs API format."""
        body: Dict[str, Any] = {"blobs": blobs}
        if metadata:
            body["metadata"] = metadata
        return self._api_request(
            f"/buckets/{self.bucket_id}/objects", body=body,
        )

    def trigger_processing(self) -> Dict[str, Any]:
        """Trigger the collection to process newly uploaded objects.

        Call this after adding texts/URLs to start feature extraction
        (embedding, OCR, face detection, etc.).

        Returns:
            Trigger response with batch status.
        """
        return self._client.collections.trigger(self.collection_id)

    # ------------------------------------------------------------------
    # Taxonomies
    # ------------------------------------------------------------------

    def create_taxonomy(
        self,
        name: str,
        config: Dict[str, Any],
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a taxonomy for classifying documents.

        Taxonomies enrich documents by matching them against retriever-based
        criteria. They run automatically during batch processing (phase 1).

        Args:
            name: Unique taxonomy name within the namespace.
            config: Taxonomy configuration dict. Must include:

                - ``taxonomy_type``: ``"flat"`` or ``"hierarchical"``
                - ``retriever_id``: Retriever that defines matching logic
                - ``collection_id``: Target collection (flat only)
                - ``input_mappings``: How document fields map to retriever inputs
                - ``enrichment_fields``: Which fields to write back

            description: Optional description.

        Returns:
            Created taxonomy with ``taxonomy_id``, ``taxonomy_name``, etc.
        """
        body: Dict[str, Any] = {
            "taxonomy_name": name,
            "config": config,
        }
        if description:
            body["description"] = description
        return self._api_request("/taxonomies", body=body)

    def list_taxonomies(self) -> Dict[str, Any]:
        """List all taxonomies in the namespace."""
        return self._api_request("/taxonomies/list", body={})

    def get_taxonomy(self, taxonomy_id: str) -> Dict[str, Any]:
        """Get a taxonomy by ID or name."""
        return self._api_request(f"/taxonomies/{taxonomy_id}", method="GET")

    def execute_taxonomy(
        self,
        taxonomy_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Test-execute a taxonomy (validation only, not production).

        Args:
            taxonomy_id: Taxonomy ID or name.

        Returns:
            Execution results with matched documents.
        """
        return self._api_request(
            f"/taxonomies/execute/{taxonomy_id}", body=kwargs or {},
        )

    def delete_taxonomy(self, taxonomy_id: str) -> Dict[str, Any]:
        """Delete a taxonomy and all its versions."""
        return self._api_request(
            f"/taxonomies/{taxonomy_id}", method="DELETE",
        )

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------

    def create_cluster(
        self,
        cluster_type: str = "vector",
        name: Optional[str] = None,
        collection_ids: Optional[List[str]] = None,
        vector_config: Optional[Dict[str, Any]] = None,
        attribute_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a cluster configuration.

        Clusters group documents by embedding similarity (vector) or
        metadata attributes. They run during batch processing (phase 2).

        Args:
            cluster_type: ``"vector"`` or ``"attribute"``.
            name: Optional cluster name.
            collection_ids: Collections to cluster. Defaults to this
                store's ``collection_id``.
            vector_config: For vector clustering — algorithm, feature_uris,
                algorithm_params (n_clusters, eps, etc.).
                Algorithms: kmeans, dbscan, hdbscan, agglomerative,
                spectral, gaussian_mixture, mean_shift, optics.
            attribute_config: For attribute clustering — attributes list,
                hierarchical_grouping, aggregation_method.

        Returns:
            Created cluster with ``cluster_id``, ``status``, etc.
        """
        body: Dict[str, Any] = {
            "collection_ids": collection_ids or [self.collection_id],
            "cluster_type": cluster_type,
        }
        if name:
            body["cluster_name"] = name
        if vector_config:
            body["vector_config"] = vector_config
        if attribute_config:
            body["attribute_config"] = attribute_config
        body.update(kwargs)
        return self._api_request("/clusters", body=body)

    def execute_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Queue a clustering job (async).

        Returns:
            Execution status with batch info.
        """
        return self._api_request(f"/clusters/{cluster_id}/execute", body={})

    def get_cluster_groups(self, cluster_id: str) -> Dict[str, Any]:
        """Get cluster groups with labels, summaries, and member counts.

        Returns:
            Dict with ``groups`` list, ``total_groups``, ``total_documents``.
        """
        return self._api_request(
            f"/clusters/{cluster_id}/groups", method="GET",
        )

    def list_clusters(self) -> Dict[str, Any]:
        """List all clusters in the namespace."""
        return self._api_request("/clusters/list", body={})

    def delete_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Delete a cluster and its execution history."""
        return self._api_request(f"/clusters/{cluster_id}", method="DELETE")

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def create_alert(
        self,
        name: str,
        retriever_id: Optional[str] = None,
        notification_config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create an alert that fires when new documents match a retriever.

        Alerts monitor document ingestion and trigger notifications
        (webhook, Slack, email) when matches are found. They run
        during batch processing (phase 3).

        Args:
            name: Alert name.
            retriever_id: Retriever that defines matching logic.
                Defaults to this store's ``retriever_id``.
            notification_config: Dict with ``channels`` list, each having
                ``channel_type`` (``"webhook"``, ``"slack"``, ``"email"``),
                ``channel_id``, and ``config``. Also ``include_matches``
                and ``include_scores`` booleans.
            enabled: Whether the alert is active.

        Returns:
            Created alert with ``alert_id``, ``name``, etc.
        """
        body: Dict[str, Any] = {
            "name": name,
            "retriever_id": retriever_id or self.retriever_id,
            "enabled": enabled,
        }
        if notification_config:
            body["notification_config"] = notification_config
        body.update(kwargs)
        return self._api_request("/alerts", body=body)

    def list_alerts(self) -> Dict[str, Any]:
        """List all alerts in the namespace."""
        return self._api_request("/alerts/list", body={})

    def get_alert(self, alert_id: str) -> Dict[str, Any]:
        """Get an alert by ID or name."""
        return self._api_request(f"/alerts/{alert_id}", method="GET")

    def get_alert_results(self, alert_id: str) -> Dict[str, Any]:
        """Get the latest execution results for an alert.

        Returns:
            Dict with ``matches`` list (document_id, score, metadata).
        """
        return self._api_request(
            f"/alerts/{alert_id}/results", method="GET",
        )

    def delete_alert(self, alert_id: str) -> Dict[str, Any]:
        """Delete an alert and its execution history."""
        return self._api_request(f"/alerts/{alert_id}", method="DELETE")

    # ------------------------------------------------------------------
    # Custom Plugins
    # ------------------------------------------------------------------

    def list_plugins(self) -> Dict[str, Any]:
        """List custom plugins deployed to this namespace.

        Returns:
            List of plugins with ``plugin_id``, ``name``, ``feature_uri``,
            ``deployment_status``, etc.
        """
        return self._api_request(
            f"/namespaces/{self.namespace}/plugins", method="GET",
        )

    def get_plugin(self, plugin_id: str) -> Dict[str, Any]:
        """Get details for a custom plugin.

        Returns:
            Plugin details including ``features`` (embedding dim, distance
            metric), ``validation_status``, ``deployment_status``.
        """
        return self._api_request(
            f"/namespaces/{self.namespace}/plugins/{plugin_id}",
            method="GET",
        )

    def get_plugin_status(self, plugin_id: str) -> Dict[str, Any]:
        """Check deployment status of a custom plugin.

        Returns:
            Dict with ``status`` (queued, pending, in_progress, deployed,
            failed), ``message``, ``estimated_completion_seconds``.
        """
        return self._api_request(
            f"/namespaces/{self.namespace}/plugins/{plugin_id}/status",
            method="GET",
        )

    def test_plugin(
        self,
        plugin_id: str,
        inputs: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Test a deployed realtime plugin with sample inputs.

        Args:
            plugin_id: Plugin ID.
            inputs: Plugin-specific input dict.
            parameters: Optional execution parameters.

        Returns:
            Dict with ``status``, ``raw_response``, ``response_type``.
        """
        body: Dict[str, Any] = {"inputs": inputs}
        if parameters:
            body["parameters"] = parameters
        return self._api_request(
            f"/namespaces/{self.namespace}/plugins/{plugin_id}/realtime/test",
            body=body,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for documents similar to the query.

        Args:
            query: Natural language search query.
            k: Maximum number of results.

        Returns:
            List of LangChain Documents with content and metadata.
        """
        response = self._client.retrievers.execute(
            retriever_id=self.retriever_id,
            inputs={"query": query},
        )
        results = response.get("results", []) if isinstance(response, dict) else response
        docs = []
        for item in results[:k]:
            metadata = dict(item.get("metadata") or {})
            for key in ("collection_id", "thumbnail_url", "_source_tier"):
                if key in item:
                    metadata[key] = item[key]
            metadata["document_id"] = item.get("document_id", "")
            metadata["score"] = item.get("score", 0.0)
            metadata["namespace"] = self.namespace

            # Extract content from metadata or top-level
            raw = metadata.pop(self.content_field, None) or item.get(self.content_field)
            if isinstance(raw, dict) and "text" in raw:
                content = str(raw["text"])
            else:
                content = str(raw) if raw else ""

            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search with relevance scores."""
        docs = self.similarity_search(query, k=k, **kwargs)
        return [(doc, doc.metadata.get("score", 0.0)) for doc in docs]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MixpeekVectorStore":
        """Create a MixpeekVectorStore and add texts.

        Requires ``api_key``, ``namespace``, ``bucket_id``,
        ``collection_id``, and ``retriever_id`` in kwargs.
        """
        store = cls(**kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    def from_retriever(
        cls,
        api_key: str,
        namespace: str,
        retriever_id: str,
        **kwargs: Any,
    ) -> "MixpeekVectorStore":
        """Create a search-only store from minimal config.

        Use this when you only need to search (no ingest). The bucket_id
        and collection_id are set to empty strings.

        Args:
            api_key: Mixpeek API key.
            namespace: Namespace to search.
            retriever_id: Retriever ID for search queries.

        Example:
            .. code-block:: python

                store = MixpeekVectorStore.from_retriever(
                    api_key="mxp_...",
                    namespace="my-ns",
                    retriever_id="ret_abc123",
                )
                docs = store.similarity_search("red cup")
        """
        return cls(
            api_key=api_key,
            namespace=namespace,
            bucket_id=kwargs.pop("bucket_id", ""),
            collection_id=kwargs.pop("collection_id", ""),
            retriever_id=retriever_id,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> "MixpeekRetriever":
        """Convert this store into a MixpeekRetriever.

        Returns a retriever configured with the same credentials.
        """
        from langchain_mixpeek.retriever import MixpeekRetriever

        return MixpeekRetriever(
            api_key=self.api_key,
            namespace=self.namespace,
            retriever_id=self.retriever_id,
            content_field=self.content_field,
            **kwargs,
        )

    def as_tool(self, **kwargs: Any) -> "BaseTool":
        """Convert this store into a MixpeekTool for agent use."""
        from langchain_mixpeek.tool import MixpeekTool

        return MixpeekTool(
            api_key=self.api_key,
            namespace=self.namespace,
            retriever_id=self.retriever_id,
            content_field=self.content_field,
            **kwargs,
        )

    def as_toolkit(self, **kwargs: Any) -> "MixpeekToolkit":
        """Convert this store into a full MixpeekToolkit for agents.

        Returns a toolkit with search, ingest, process, classify,
        cluster, and alert tools.
        """
        from langchain_mixpeek.toolkit import MixpeekToolkit

        return MixpeekToolkit(store=self, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents by ID."""
        if ids:
            for doc_id in ids:
                self._client.documents.delete(doc_id)
