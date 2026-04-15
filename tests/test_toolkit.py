"""Unit tests for MixpeekToolkit and individual tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from langchain_mixpeek import (
    MixpeekAlertTool,
    MixpeekClassifyTool,
    MixpeekClusterTool,
    MixpeekIngestTool,
    MixpeekProcessTool,
    MixpeekSearchTool,
    MixpeekToolkit,
    MixpeekVectorStore,
)


def _mock_mixpeek():
    mock_client = MagicMock()
    mock_client.retrievers.execute.return_value = {
        "results": [
            {
                "document_id": "doc_001",
                "score": 0.95,
                "text": "Bold camo pattern.",
                "collection_id": "col_test",
            },
        ],
        "status": "completed",
    }
    mock_client.collections.trigger.return_value = {"batch_id": "batch_abc", "status": "processing"}
    return patch("langchain_mixpeek.vectorstore.Mixpeek", return_value=mock_client), mock_client


def _make_store(**kwargs):
    defaults = dict(
        api_key="mxp_test",
        namespace="test-ns",
        bucket_id="bkt_test",
        collection_id="col_test",
        retriever_id="ret_test",
    )
    defaults.update(kwargs)
    return MixpeekVectorStore(**defaults)


def _mock_api_request(return_value=None):
    result = return_value or {}
    return patch.object(MixpeekVectorStore, "_api_request", return_value=result)


# ---------------------------------------------------------------------------
# MixpeekToolkit tests
# ---------------------------------------------------------------------------


class TestMixpeekToolkit:
    def test_get_tools_returns_all_six(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            toolkit = MixpeekToolkit(
                api_key="mxp_test",
                namespace="test-ns",
                bucket_id="bkt_test",
                collection_id="col_test",
                retriever_id="ret_test",
            )
            tools = toolkit.get_tools()

        assert len(tools) == 6
        names = {t.name for t in tools}
        assert names == {
            "mixpeek_search",
            "mixpeek_ingest",
            "mixpeek_process",
            "mixpeek_classify",
            "mixpeek_cluster",
            "mixpeek_alert",
        }

    def test_get_tools_with_actions_filter(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            toolkit = MixpeekToolkit(
                api_key="mxp_test",
                namespace="test-ns",
                bucket_id="bkt_test",
                collection_id="col_test",
                retriever_id="ret_test",
            )
            tools = toolkit.get_tools(actions=["search", "ingest"])

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"mixpeek_search", "mixpeek_ingest"}

    def test_toolkit_from_existing_store(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
            toolkit = MixpeekToolkit(store=store)

        assert toolkit.store is store
        assert len(toolkit.get_tools()) == 6

    def test_store_property(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
            toolkit = MixpeekToolkit(store=store)

        assert toolkit.store.api_key == "mxp_test"


# ---------------------------------------------------------------------------
# Individual tool tests
# ---------------------------------------------------------------------------


class TestMixpeekSearchTool:
    def test_search_returns_json(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
            tool = MixpeekSearchTool(store=store)
            result = tool.invoke("camo pattern")

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["document_id"] == "doc_001"
        assert parsed[0]["score"] == 0.95
        assert parsed[0]["content"] == "Bold camo pattern."

    def test_search_error_returns_message(self):
        patcher, mock_client = _mock_mixpeek()
        mock_client.retrievers.execute.side_effect = Exception("API down")
        with patcher:
            store = _make_store()
            tool = MixpeekSearchTool(store=store)
            result = tool.invoke("test")

        assert "Search failed" in result


class TestMixpeekIngestTool:
    def test_ingest_text(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = patch.object(
            MixpeekVectorStore, "_upload_object",
            return_value={"object_id": "obj_123"},
        )
        with patcher, upload_patcher:
            store = _make_store()
            tool = MixpeekIngestTool(store=store)
            result = tool.invoke(json.dumps({
                "type": "text",
                "data": "hello world",
                "metadata": {"source": "test"},
            }))

        parsed = json.loads(result)
        assert parsed["object_ids"] == ["obj_123"]
        assert parsed["status"] == "uploaded"

    def test_ingest_url(self):
        patcher, _ = _mock_mixpeek()
        upload_patcher = patch.object(
            MixpeekVectorStore, "_upload_object",
            return_value={"object_id": "obj_456"},
        )
        with patcher, upload_patcher:
            store = _make_store()
            tool = MixpeekIngestTool(store=store)
            result = tool.invoke(json.dumps({
                "type": "video",
                "url": "https://example.com/clip.mp4",
            }))

        parsed = json.loads(result)
        assert parsed["object_ids"] == ["obj_456"]


class TestMixpeekProcessTool:
    def test_trigger(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            store = _make_store()
            tool = MixpeekProcessTool(store=store)
            result = tool.invoke("")

        parsed = json.loads(result)
        assert parsed["status"] == "processing"


class TestMixpeekClassifyTool:
    def test_execute_taxonomy(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"results": [{"label": "streetwear"}]})
        with patcher, api_patcher:
            store = _make_store()
            tool = MixpeekClassifyTool(store=store)
            result = tool.invoke(json.dumps({"taxonomy_id": "tax_123"}))

        parsed = json.loads(result)
        assert parsed["results"][0]["label"] == "streetwear"


class TestMixpeekClusterTool:
    def test_create_and_execute(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"cluster_id": "cls_123"})
        with patcher, api_patcher:
            store = _make_store()
            tool = MixpeekClusterTool(store=store)
            result = tool.invoke(json.dumps({
                "cluster_type": "vector",
                "algorithm": "kmeans",
                "n_clusters": 3,
            }))

        parsed = json.loads(result)
        assert parsed["cluster_id"] == "cls_123"
        assert parsed["status"] == "executing"

    def test_get_existing_groups(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({
            "groups": [{"label": "Group A", "member_count": 10}],
            "total_groups": 1,
        })
        with patcher, api_patcher:
            store = _make_store()
            tool = MixpeekClusterTool(store=store)
            result = tool.invoke(json.dumps({"cluster_id": "cls_123"}))

        parsed = json.loads(result)
        assert parsed["total_groups"] == 1


class TestMixpeekAlertTool:
    def test_create_webhook_alert(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"alert_id": "alt_123", "name": "brand-watch"})
        with patcher, api_patcher:
            store = _make_store()
            tool = MixpeekAlertTool(store=store)
            result = tool.invoke(json.dumps({
                "name": "brand-watch",
                "webhook_url": "https://example.com/hook",
            }))

        parsed = json.loads(result)
        assert parsed["alert_id"] == "alt_123"

    def test_get_alert_results(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({
            "matches": [{"document_id": "doc_1", "score": 0.92}],
        })
        with patcher, api_patcher:
            store = _make_store()
            tool = MixpeekAlertTool(store=store)
            result = tool.invoke(json.dumps({"alert_id": "alt_123"}))

        parsed = json.loads(result)
        assert len(parsed["matches"]) == 1

    def test_create_multi_channel_alert(self):
        patcher, _ = _mock_mixpeek()
        api_patcher = _mock_api_request({"alert_id": "alt_456"})
        with patcher, api_patcher as mock_api:
            store = _make_store()
            tool = MixpeekAlertTool(store=store)
            tool.invoke(json.dumps({
                "name": "multi-alert",
                "webhook_url": "https://example.com/hook",
                "slack_channel": "#alerts",
                "email": "team@example.com",
            }))

        # Verify 3 channels were configured
        call_body = mock_api.call_args[1]["body"]
        channels = call_body["notification_config"]["channels"]
        assert len(channels) == 3
        types = {c["channel_type"] for c in channels}
        assert types == {"webhook", "slack", "email"}
