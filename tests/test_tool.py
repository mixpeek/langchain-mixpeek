"""Unit tests for MixpeekTool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from langchain_mixpeek import MixpeekTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_API_RESULTS = [
    {
        "document_id": "doc_001",
        "score": 0.95,
        "text": "The red cup is on the table.",
        "metadata": {
            "transcript_chunk": "The red cup is on the table.",
            "source_url": "https://example.com/video.mp4",
            "timestamp": "00:01:23",
        },
    },
    {
        "document_id": "doc_002",
        "score": 0.80,
        "text": "A red cup sits near the window.",
        "metadata": {
            "transcript_chunk": "A red cup sits near the window.",
            "source_url": "https://example.com/video2.mp4",
            "timestamp": "00:02:10",
        },
    },
    {
        "document_id": "doc_003",
        "score": 0.65,
        "text": "Someone picks up the red cup.",
        "metadata": {
            "transcript_chunk": "Someone picks up the red cup.",
            "source_url": "https://example.com/video3.mp4",
            "timestamp": "00:03:45",
        },
    },
]


def _make_tool(**kwargs) -> MixpeekTool:
    defaults = dict(
        api_key="mxp_test",
        retriever_id="ret_test123",
        namespace="test-ns",
    )
    defaults.update(kwargs)
    return MixpeekTool(**defaults)


def _mock_mixpeek(results=None):
    if results is None:
        results = FAKE_API_RESULTS
    mock_client = MagicMock()
    # SDK returns {"results": [...], "status": "completed", ...}
    mock_client.retrievers.execute.return_value = {
        "results": results,
        "status": "completed",
        "execution_id": "exec_test",
        "total_count": len(results),
    }
    return patch("langchain_mixpeek.tool.Mixpeek", return_value=mock_client, ), mock_client


# ---------------------------------------------------------------------------
# MixpeekTool tests
# ---------------------------------------------------------------------------


class TestMixpeekTool:
    def test_run_returns_json_string(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            tool = _make_tool()
            result = tool.invoke("find the red cup")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_result_fields_present(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            tool = _make_tool()
            result = tool.invoke("red cup")

        parsed = json.loads(result)
        first = parsed[0]
        assert "document_id" in first
        assert "score" in first
        assert "source" in first
        assert "timestamp" in first
        assert "content" in first

    def test_document_id_correct(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            tool = _make_tool()
            result = tool.invoke("red cup")

        parsed = json.loads(result)
        assert parsed[0]["document_id"] == "doc_001"
        assert parsed[1]["document_id"] == "doc_002"

    def test_score_rounded(self):
        results = [
            {"document_id": "x", "score": 0.123456789, "metadata": {}}
        ]
        patcher, _ = _mock_mixpeek(results=results)
        with patcher:
            tool = _make_tool()
            result = tool.invoke("anything")

        parsed = json.loads(result)
        assert parsed[0]["score"] == 0.1235

    def test_source_from_source_url(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            tool = _make_tool()
            result = tool.invoke("red cup")

        parsed = json.loads(result)
        assert parsed[0]["source"] == "https://example.com/video.mp4"

    def test_source_fallback_to_file_name(self):
        results = [
            {
                "document_id": "doc_a",
                "score": 0.7,
                "metadata": {"file_name": "clip.mp4"},
            }
        ]
        patcher, _ = _mock_mixpeek(results=results)
        with patcher:
            tool = _make_tool()
            result = tool.invoke("anything")

        parsed = json.loads(result)
        assert parsed[0]["source"] == "clip.mp4"

    def test_content_uses_content_field(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            tool = _make_tool(content_field="transcript_chunk")
            result = tool.invoke("red cup")

        parsed = json.loads(result)
        assert parsed[0]["content"] == "The red cup is on the table."

    def test_top_k_limits_results(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            tool = _make_tool(top_k=2)
            result = tool.invoke("red cup")

        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_top_k_zero_returns_empty_list(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            tool = _make_tool(top_k=0)
            result = tool.invoke("red cup")

        parsed = json.loads(result)
        assert parsed == []

    def test_query_forwarded_to_api(self):
        patcher, mock_client = _mock_mixpeek()
        with patcher:
            tool = _make_tool()
            tool.invoke("unique query string")

        call_kwargs = mock_client.retrievers.execute.call_args.kwargs
        assert call_kwargs["inputs"]["query"] == "unique query string"
        assert call_kwargs["retriever_id"] == "ret_test123"

    def test_exception_returns_error_string(self):
        mock_client = MagicMock()
        mock_client.retrievers.execute.side_effect = RuntimeError("API timeout")
        with patch("langchain_mixpeek.tool.Mixpeek", return_value=mock_client, ):
            tool = _make_tool()
            result = tool.invoke("anything")

        assert "Search failed" in result
        assert "API timeout" in result

    def test_exception_result_is_not_json(self):
        """Error strings should NOT be valid JSON (they're plain text)."""
        mock_client = MagicMock()
        mock_client.retrievers.execute.side_effect = ValueError("bad request")
        with patch("langchain_mixpeek.tool.Mixpeek", return_value=mock_client, ):
            tool = _make_tool()
            result = tool.invoke("anything")

        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    def test_none_metadata_handled(self):
        results = [{"document_id": "doc_none", "score": 0.5, "metadata": None}]
        patcher, _ = _mock_mixpeek(results=results)
        with patcher:
            tool = _make_tool()
            result = tool.invoke("anything")

        parsed = json.loads(result)
        assert parsed[0]["document_id"] == "doc_none"
        assert parsed[0]["source"] == ""
        assert parsed[0]["content"] is None or parsed[0]["content"] == ""

    def test_tool_name_and_description(self):
        tool = _make_tool()
        assert tool.name == "mixpeek_search"
        assert "video" in tool.description.lower()
        assert "natural language" in tool.description.lower()

    def test_return_direct_false_by_default(self):
        tool = _make_tool()
        assert tool.return_direct is False

    def test_empty_results_returns_empty_json_array(self):
        patcher, _ = _mock_mixpeek(results=[])
        with patcher:
            tool = _make_tool()
            result = tool.invoke("anything")

        assert json.loads(result) == []
