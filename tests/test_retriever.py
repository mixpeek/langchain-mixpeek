"""Unit tests for MixpeekRetriever and AsyncMixpeekRetriever."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from langchain_mixpeek import AsyncMixpeekRetriever, MixpeekRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_API_RESULTS = [
    {
        "document_id": "doc_001",
        "score": 0.95,
        "metadata": {
            "transcript_chunk": "The red cup is on the table.",
            "source_url": "https://example.com/video.mp4",
            "timestamp": "00:01:23",
        },
    },
    {
        "document_id": "doc_002",
        "score": 0.80,
        "metadata": {
            "transcript_chunk": "A red cup sits near the window.",
            "source_url": "https://example.com/video2.mp4",
            "timestamp": "00:02:10",
        },
    },
    {
        "document_id": "doc_003",
        "score": 0.65,
        "metadata": {
            "transcript_chunk": "Someone picks up the red cup.",
            "source_url": "https://example.com/video3.mp4",
            "timestamp": "00:03:45",
        },
    },
]


def _make_retriever(**kwargs) -> MixpeekRetriever:
    defaults = dict(
        api_key="mxp_test",
        retriever_id="ret_test123",
        namespace="test-ns",
    )
    defaults.update(kwargs)
    return MixpeekRetriever(**defaults)


def _mock_mixpeek(results=None):
    """Return a context-manager patch that stubs out the Mixpeek client."""
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
    return patch("langchain_mixpeek.retriever.Mixpeek", return_value=mock_client, ), mock_client


# ---------------------------------------------------------------------------
# MixpeekRetriever tests
# ---------------------------------------------------------------------------


class TestMixpeekRetriever:
    def test_returns_documents(self):
        patcher, mock_client = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever()
            docs = retriever.invoke("find the red cup")

        assert len(docs) == 3
        for doc in docs:
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")

    def test_content_field_used_as_page_content(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever(content_field="transcript_chunk")
            docs = retriever.invoke("red cup")

        assert docs[0].page_content == "The red cup is on the table."
        assert docs[1].page_content == "A red cup sits near the window."

    def test_content_field_not_in_metadata(self):
        """content_field should be popped from metadata, not duplicated."""
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever(content_field="transcript_chunk")
            docs = retriever.invoke("red cup")

        for doc in docs:
            assert "transcript_chunk" not in doc.metadata

    def test_document_id_in_metadata(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever()
            docs = retriever.invoke("red cup")

        assert docs[0].metadata["document_id"] == "doc_001"
        assert docs[1].metadata["document_id"] == "doc_002"

    def test_score_in_metadata(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever()
            docs = retriever.invoke("red cup")

        assert docs[0].metadata["score"] == 0.95
        assert docs[1].metadata["score"] == 0.80

    def test_namespace_in_metadata(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever(namespace="prod-ns")
            docs = retriever.invoke("red cup")

        for doc in docs:
            assert doc.metadata["namespace"] == "prod-ns"

    def test_top_k_limits_results(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever(top_k=2)
            docs = retriever.invoke("red cup")

        assert len(docs) == 2

    def test_top_k_zero_returns_empty(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever(top_k=0)
            docs = retriever.invoke("red cup")

        assert docs == []

    def test_filters_passed_to_api(self):
        patcher, mock_client = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever(filters={"category": "sports"})
            retriever.invoke("red cup")

        call_kwargs = mock_client.retrievers.execute.call_args.kwargs
        assert call_kwargs["inputs"]["filters"] == {"category": "sports"}

    def test_no_filters_by_default(self):
        patcher, mock_client = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever()
            retriever.invoke("red cup")

        call_kwargs = mock_client.retrievers.execute.call_args.kwargs
        assert "filters" not in call_kwargs["inputs"]

    def test_empty_results(self):
        patcher, _ = _mock_mixpeek(results=[])
        with patcher:
            retriever = _make_retriever()
            docs = retriever.invoke("red cup")

        assert docs == []

    def test_missing_content_field_becomes_empty_string(self):
        results = [
            {
                "document_id": "doc_x",
                "score": 0.5,
                "metadata": {"source_url": "https://example.com"},
                # no transcript_chunk
            }
        ]
        patcher, _ = _mock_mixpeek(results=results)
        with patcher:
            retriever = _make_retriever(content_field="transcript_chunk")
            docs = retriever.invoke("anything")

        assert docs[0].page_content == ""

    def test_none_metadata_handled(self):
        results = [{"document_id": "doc_y", "score": 0.3, "metadata": None}]
        patcher, _ = _mock_mixpeek(results=results)
        with patcher:
            retriever = _make_retriever()
            docs = retriever.invoke("anything")

        assert docs[0].metadata["document_id"] == "doc_y"

    def test_query_forwarded_to_api(self):
        patcher, mock_client = _mock_mixpeek()
        with patcher:
            retriever = _make_retriever()
            retriever.invoke("find the red cup")

        call_kwargs = mock_client.retrievers.execute.call_args.kwargs
        assert call_kwargs["inputs"]["query"] == "find the red cup"
        assert call_kwargs["retriever_id"] == "ret_test123"

    def test_custom_content_field(self):
        results = [
            {
                "document_id": "doc_z",
                "score": 0.9,
                "metadata": {"caption": "A sunny day.", "transcript_chunk": "ignored"},
            }
        ]
        patcher, _ = _mock_mixpeek(results=results)
        with patcher:
            retriever = _make_retriever(content_field="caption")
            docs = retriever.invoke("sunny")

        assert docs[0].page_content == "A sunny day."
        assert "caption" not in docs[0].metadata


# ---------------------------------------------------------------------------
# AsyncMixpeekRetriever tests
# ---------------------------------------------------------------------------


class TestAsyncMixpeekRetriever:
    def test_async_retriever_is_subclass(self):
        assert issubclass(AsyncMixpeekRetriever, MixpeekRetriever)

    def test_async_invoke_returns_documents(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = AsyncMixpeekRetriever(
                api_key="mxp_test",
                retriever_id="ret_test123",
                namespace="test-ns",
            )
            docs = asyncio.get_event_loop().run_until_complete(
                retriever.ainvoke("find the red cup")
            )

        assert len(docs) == 3
        assert docs[0].page_content == "The red cup is on the table."

    def test_async_top_k(self):
        patcher, _ = _mock_mixpeek()
        with patcher:
            retriever = AsyncMixpeekRetriever(
                api_key="mxp_test",
                retriever_id="ret_test123",
                namespace="test-ns",
                top_k=1,
            )
            docs = asyncio.get_event_loop().run_until_complete(
                retriever.ainvoke("red cup")
            )

        assert len(docs) == 1
