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


def _mock_api_request(return_value=None):
    """Mock the _api_request method on MixpeekVectorStore."""
    result = return_value or {}
    return patch.object(MixpeekVectorStore, "_api_request", return_value=result)


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


class TestTaxonomies:
    def test_create_taxonomy(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"taxonomy_id": "tax_123", "taxonomy_name": "brands"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.create_taxonomy(
                name="brands",
                config={"taxonomy_type": "flat", "retriever_id": "ret_test", "collection_id": "col_test"},
                description="Brand classifier",
            )

        assert result["taxonomy_id"] == "tax_123"
        call = mock_api.call_args
        assert call[0][0] == "/taxonomies"
        body = call[1]["body"]
        assert body["taxonomy_name"] == "brands"
        assert body["description"] == "Brand classifier"

    def test_list_taxonomies(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"taxonomies": []})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.list_taxonomies()

        assert "taxonomies" in result
        mock_api.assert_called_once_with("/taxonomies/list", body={})

    def test_get_taxonomy(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"taxonomy_id": "tax_123"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.get_taxonomy("tax_123")

        mock_api.assert_called_once_with("/taxonomies/tax_123", method="GET")

    def test_execute_taxonomy(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"results": []})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.execute_taxonomy("tax_123")

        mock_api.assert_called_once_with("/taxonomies/execute/tax_123", body={})

    def test_delete_taxonomy(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"status": "deleted"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.delete_taxonomy("tax_123")

        mock_api.assert_called_once_with("/taxonomies/tax_123", method="DELETE")


class TestClusters:
    def test_create_cluster_defaults_to_store_collection(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"cluster_id": "cls_123"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.create_cluster(
                cluster_type="vector",
                vector_config={"algorithm": "kmeans", "algorithm_params": {"n_clusters": 5}},
            )

        assert result["cluster_id"] == "cls_123"
        body = mock_api.call_args[1].get("body") or mock_api.call_args[0][1]
        assert body["collection_ids"] == ["col_test"]
        assert body["cluster_type"] == "vector"

    def test_execute_cluster(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"status": "running"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.execute_cluster("cls_123")

        mock_api.assert_called_once_with("/clusters/cls_123/execute", body={})

    def test_get_cluster_groups(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"groups": [], "total_groups": 0})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.get_cluster_groups("cls_123")

        assert result["total_groups"] == 0
        mock_api.assert_called_once_with("/clusters/cls_123/groups", method="GET")

    def test_list_clusters(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"clusters": []})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.list_clusters()

        mock_api.assert_called_once_with("/clusters/list", body={})

    def test_delete_cluster(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"status": "deleted"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.delete_cluster("cls_123")

        mock_api.assert_called_once_with("/clusters/cls_123", method="DELETE")


class TestAlerts:
    def test_create_alert_defaults_to_store_retriever(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"alert_id": "alt_123", "name": "brand-match"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.create_alert(
                name="brand-match",
                notification_config={
                    "channels": [{"channel_type": "webhook", "config": {"url": "https://example.com/hook"}}],
                    "include_matches": True,
                    "include_scores": True,
                },
            )

        assert result["alert_id"] == "alt_123"
        body = mock_api.call_args[1].get("body") or mock_api.call_args[0][1]
        assert body["retriever_id"] == "ret_test"
        assert body["name"] == "brand-match"

    def test_get_alert_results(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"matches": [{"document_id": "doc_1", "score": 0.9}]})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.get_alert_results("alt_123")

        assert len(result["matches"]) == 1
        mock_api.assert_called_once_with("/alerts/alt_123/results", method="GET")

    def test_list_alerts(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"alerts": []})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.list_alerts()

        mock_api.assert_called_once_with("/alerts/list", body={})

    def test_delete_alert(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"status": "deleted"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.delete_alert("alt_123")

        mock_api.assert_called_once_with("/alerts/alt_123", method="DELETE")


class TestPlugins:
    def test_list_plugins(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request([{"plugin_id": "plg_1", "name": "my_extractor"}])
        with patcher, api_patcher as mock_api:
            store = _make_store()
            store.list_plugins()

        mock_api.assert_called_once_with("/namespaces/test-ns/plugins", method="GET")

    def test_get_plugin(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"plugin_id": "plg_1", "deployment_status": "deployed"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.get_plugin("plg_1")

        assert result["deployment_status"] == "deployed"
        mock_api.assert_called_once_with("/namespaces/test-ns/plugins/plg_1", method="GET")

    def test_get_plugin_status(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"status": "deployed", "message": "Running"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.get_plugin_status("plg_1")

        assert result["status"] == "deployed"
        mock_api.assert_called_once_with("/namespaces/test-ns/plugins/plg_1/status", method="GET")

    def test_test_plugin(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"status": "success", "raw_response": {"embedding": [0.1]}})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            result = store.test_plugin("plg_1", inputs={"text": "hello"})

        assert result["status"] == "success"
        body = mock_api.call_args[1].get("body") or mock_api.call_args[0][1]
        assert body["inputs"] == {"text": "hello"}


class TestConversions:
    def test_from_retriever(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = MixpeekVectorStore.from_retriever(
                api_key="mxp_test",
                namespace="test-ns",
                retriever_id="ret_test",
            )

        assert store.api_key == "mxp_test"
        assert store.retriever_id == "ret_test"
        assert store.bucket_id == ""
        assert store.collection_id == ""

    def test_as_retriever(self):
        from langchain_mixpeek import MixpeekRetriever

        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
            retriever = store.as_retriever()

        assert isinstance(retriever, MixpeekRetriever)
        assert retriever.api_key == "mxp_test"
        assert retriever.retriever_id == "ret_test"
        assert retriever.namespace == "test-ns"

    def test_as_tool(self):
        from langchain_mixpeek import MixpeekTool

        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
            tool = store.as_tool()

        assert isinstance(tool, MixpeekTool)
        assert tool.api_key == "mxp_test"
        assert tool.retriever_id == "ret_test"

    def test_as_toolkit(self):
        from langchain_mixpeek import MixpeekToolkit

        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
            toolkit = store.as_toolkit()

        assert isinstance(toolkit, MixpeekToolkit)
        assert len(toolkit.get_tools()) == 6
        assert toolkit.store is store
