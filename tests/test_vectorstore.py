"""Unit tests for MixpeekVectorStore."""

import json
from unittest.mock import MagicMock, patch

import pytest

from langchain_mixpeek import MixpeekVectorStore


FAKE_API_RESULTS = [
    {
        "document_id": "doc_001",
        "score": 0.95,
        "trend_insight": {"text": "Bold camo pattern.", "token_usage": 10},
    },
    {
        "document_id": "doc_002",
        "score": 0.80,
        "trend_insight": {"text": "Streetwear graphics.", "token_usage": 8},
    },
]


def _make_store(**kwargs) -> MixpeekVectorStore:
    defaults = dict(
        api_key="mxp_test",
        namespace="test-ns",
        bucket_id="bkt_test",
        collection_id="col_test",
        retriever_id="ret_test",
    )
    defaults.update(kwargs)
    return MixpeekVectorStore(**defaults)


def _mock_mixpeek(search_results=None, upload_result=None):
    if search_results is None:
        search_results = FAKE_API_RESULTS
    mock_client = MagicMock()
    mock_client.retrievers.execute.return_value = {
        "results": search_results,
        "status": "completed",
    }
    mock_client.collections.trigger.return_value = {"batch_id": "batch_abc", "status": "processing"}
    mock_client.documents.delete.return_value = {"status": "deleted"}
    return patch("langchain_mixpeek.vectorstore.Mixpeek", return_value=mock_client), mock_client


def _mock_upload(upload_result=None):
    """Mock the _upload_object method on MixpeekVectorStore."""
    result = upload_result or {"object_id": "obj_123"}
    return patch.object(MixpeekVectorStore, "_upload_object", return_value=result)


class TestMixpeekVectorStore:
    def test_similarity_search_returns_documents(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store(content_field="trend_insight")
            docs = store.similarity_search("camo", k=2)

        assert len(docs) == 2
        assert docs[0].page_content == "Bold camo pattern."
        assert docs[0].metadata["document_id"] == "doc_001"
        assert docs[0].metadata["score"] == 0.95

    def test_similarity_search_with_score(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store(content_field="trend_insight")
            results = store.similarity_search_with_score("camo", k=2)

        assert len(results) == 2
        doc, score = results[0]
        assert score == 0.95
        assert doc.page_content == "Bold camo pattern."

    def test_similarity_search_k_limits(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store(content_field="trend_insight")
            docs = store.similarity_search("camo", k=1)

        assert len(docs) == 1

    def test_add_texts(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_texts(["hello", "world"], metadatas=[{"a": 1}, {"b": 2}])

        assert len(ids) == 2
        assert ids[0] == "obj_123"
        assert mock_upload.call_count == 2
        call1 = mock_upload.call_args_list[0]
        assert call1[1]["blobs"][0]["data"] == "hello"
        assert call1[1]["blobs"][0]["type"] == "text"
        assert call1[1]["metadata"] == {"a": 1}

    def test_add_urls(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_urls(
                ["https://example.com/video.mp4"],
                metadatas=[{"source": "test"}],
            )

        assert len(ids) == 1
        call = mock_upload.call_args
        assert call[1]["blobs"][0]["url"] == "https://example.com/video.mp4"
        assert call[1]["metadata"] == {"source": "test"}

    def test_trigger_processing(self):
        patcher, mock_client = _mock_mixpeek()
        with patcher:
            store = _make_store()
            result = store.trigger_processing()

        assert result["status"] == "processing"
        mock_client.collections.trigger.assert_called_once_with("col_test")

    def test_delete(self):
        patcher, mock_client = _mock_mixpeek()
        with patcher:
            store = _make_store()
            store.delete(ids=["doc_001", "doc_002"])

        assert mock_client.documents.delete.call_count == 2

    def test_embeddings_is_none(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
        assert store.embeddings is None

    def test_from_texts(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = MixpeekVectorStore.from_texts(
                ["test"],
                api_key="mxp_test",
                namespace="test-ns",
                bucket_id="bkt_test",
                collection_id="col_test",
                retriever_id="ret_test",
            )

        assert mock_upload.call_count == 1

    def test_empty_search_results(self):
        patcher, _ = _mock_mixpeek(search_results=[])
        with patcher:
            store = _make_store()
            docs = store.similarity_search("nothing")

        assert docs == []

    def test_add_images(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_images(["https://example.com/photo.jpg"])

        assert len(ids) == 1
        assert mock_upload.call_args[1]["blobs"][0]["type"] == "image"
        assert mock_upload.call_args[1]["blobs"][0]["url"] == "https://example.com/photo.jpg"

    def test_add_videos(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_videos(["https://example.com/clip.mp4"])

        assert len(ids) == 1
        assert mock_upload.call_args[1]["blobs"][0]["type"] == "video"

    def test_add_audio(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_audio(["https://example.com/song.mp3"])

        assert len(ids) == 1
        assert mock_upload.call_args[1]["blobs"][0]["type"] == "audio"

    def test_add_pdfs(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_pdfs(["https://example.com/doc.pdf"])

        assert len(ids) == 1
        assert mock_upload.call_args[1]["blobs"][0]["type"] == "pdf"

    def test_add_excel(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_excel(["https://example.com/data.xlsx"])

        assert len(ids) == 1
        assert mock_upload.call_args[1]["blobs"][0]["type"] == "excel"

    def test_add_videos_with_metadata(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = _mock_upload()
        with patcher, upload_patcher as mock_upload:
            store = _make_store()
            ids = store.add_videos(
                ["https://example.com/a.mp4", "https://example.com/b.mp4"],
                metadatas=[{"title": "A"}, {"title": "B"}],
            )

        assert len(ids) == 2
        assert mock_upload.call_args_list[0][1]["metadata"] == {"title": "A"}
        assert mock_upload.call_args_list[1][1]["metadata"] == {"title": "B"}
