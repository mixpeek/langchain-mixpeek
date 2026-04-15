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

    def _upload_object(
        self,
        blobs: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Upload an object to the bucket using the blobs API format."""
        body: Dict[str, Any] = {"blobs": blobs}
        if metadata:
            body["metadata"] = metadata
        req = urllib.request.Request(
            f"{self._base_url}/buckets/{self.bucket_id}/objects",
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Namespace": self.namespace,
            },
            data=json.dumps(body).encode(),
        )
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read().decode())

    def trigger_processing(self) -> Dict[str, Any]:
        """Trigger the collection to process newly uploaded objects.

        Call this after adding texts/URLs to start feature extraction
        (embedding, OCR, face detection, etc.).

        Returns:
            Trigger response with batch status.
        """
        return self._client.collections.trigger(self.collection_id)

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

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents by ID."""
        if ids:
            for doc_id in ids:
                self._client.documents.delete(doc_id)
