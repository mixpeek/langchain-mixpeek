"""
Agents Are Blind — Mixpeek Gives Them Eyes

This demo shows a LangChain agent using MixpeekTool to search across
multimodal content (images, video, audio, documents) and reason over
the results. Without Mixpeek, the agent has no way to "see" or "hear"
unstructured media. With it, the agent can answer questions about
visual content, find specific moments in video, and analyze brand assets.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/agent_demo.py

Requires:
    pip install langchain-mixpeek langchain-openai langgraph
"""

import os
import sys

from langchain_core.prompts import ChatPromptTemplate

from langchain_mixpeek import MixpeekRetriever, MixpeekTool

# ---------------------------------------------------------------------------
# Config — swap these for your own namespace/retriever
# ---------------------------------------------------------------------------
MIXPEEK_API_KEY = os.environ.get(
    "MIXPEEK_API_KEY",
    "mxp_sk_gkhp2UTrzWIbgTponASAqBdiG42xePPxTrp19jpHjQvUi98qc86G5y20wGpN5I3wX84",
)
RETRIEVER_ID = "ret_039d833935b255"  # bape-trend-matching-v5-fast
NAMESPACE = "bape-brand-brain"


def _get_llm():
    """Return the best available LLM, or None if no API key is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return None


def run_retriever_e2e():
    """Bare retriever E2E — no LLM, just verify Mixpeek returns docs."""
    print("=" * 60)
    print("DEMO 0: Raw Retriever E2E (no LLM)")
    print("=" * 60)

    retriever = MixpeekRetriever(
        api_key=MIXPEEK_API_KEY,
        retriever_id=RETRIEVER_ID,
        namespace=NAMESPACE,
        top_k=3,
        content_field="trend_insight",
    )

    docs = retriever.invoke("camo pattern")
    print(f"\nQuery: 'camo pattern' → {len(docs)} documents\n")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] score={doc.metadata.get('score', 'n/a')}")
        print(f"      doc_id={doc.metadata.get('document_id', 'n/a')}")
        content = doc.page_content[:200] if doc.page_content else "(no content)"
        print(f"      content={content}")
        print()


def run_retriever_chain():
    """RAG chain: retriever feeds context to LLM."""
    print("=" * 60)
    print("DEMO 1: Retriever → RAG Chain")
    print("=" * 60)

    # Pick an available LLM
    llm = _get_llm()
    if llm is None:
        print("\nSkipped — no LLM API key available (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return

    retriever = MixpeekRetriever(
        api_key=MIXPEEK_API_KEY,
        retriever_id=RETRIEVER_ID,
        namespace=NAMESPACE,
        top_k=5,
        content_field="trend_insight",
    )

    prompt = ChatPromptTemplate.from_template(
        "You are a brand analyst. Use the search results below to answer the question.\n\n"
        "Search results:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely, citing specific items when relevant."
    )

    chain = {"context": retriever, "question": lambda x: x} | prompt | llm

    question = "What streetwear trends connect to BAPE's archive?"
    print(f"\nQuestion: {question}")
    print("-" * 40)
    response = chain.invoke(question)
    print(f"Answer: {response.content}\n")


def run_tool_e2e():
    """Tool E2E — verify MixpeekTool returns structured JSON for agents."""
    print("=" * 60)
    print("DEMO 1.5: MixpeekTool E2E (no LLM)")
    print("=" * 60)

    import json

    trend_tool = MixpeekTool(
        api_key=MIXPEEK_API_KEY,
        retriever_id="ret_039d833935b255",
        namespace=NAMESPACE,
        top_k=3,
        name="search_trends",
        description="Search BAPE's visual archive for streetwear trend connections.",
        content_field="trend_insight",
    )

    brand_tool = MixpeekTool(
        api_key=MIXPEEK_API_KEY,
        retriever_id="ret_b990bf1f7851f9",
        namespace=NAMESPACE,
        top_k=3,
        name="check_brand_alignment",
        description="Check whether items align with a specific style or concept.",
        content_field="brand_alignment",
    )

    print("\n--- search_trends('bold graphics') ---")
    result = trend_tool.invoke("bold graphics")
    parsed = json.loads(result)
    for r in parsed:
        print(f"  score={r['score']} | {r['content'][:150]}")

    print("\n--- check_brand_alignment('minimalist design') ---")
    result = brand_tool.invoke("minimalist design")
    parsed = json.loads(result)
    for r in parsed:
        print(f"  score={r['score']} | {r['content'][:150]}")
    print()


def run_agent():
    """ReAct agent with MixpeekTool — the agent decides when to search."""
    print("=" * 60)
    print("DEMO 2: ReAct Agent with MixpeekTool")
    print("=" * 60)

    llm = _get_llm()
    if llm is None:
        print("\nSkipped — no LLM API key available (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return

    from langgraph.prebuilt import create_react_agent

    trend_tool = MixpeekTool(
        api_key=MIXPEEK_API_KEY,
        retriever_id="ret_039d833935b255",
        namespace=NAMESPACE,
        top_k=3,
        name="search_trends",
        description=(
            "Search BAPE's visual archive for streetwear trend connections. "
            "Input: natural language query about fashion trends, styles, or aesthetics."
        ),
        content_field="trend_insight",
    )

    brand_tool = MixpeekTool(
        api_key=MIXPEEK_API_KEY,
        retriever_id="ret_b990bf1f7851f9",
        namespace=NAMESPACE,
        top_k=3,
        name="check_brand_alignment",
        description=(
            "Check whether items in BAPE's archive align with a specific style or concept. "
            "Input: natural language description of a style, product, or concept to check against."
        ),
        content_field="brand_alignment",
    )

    tools = [trend_tool, brand_tool]

    agent = create_react_agent(
        llm,
        tools,
        prompt=(
            "You are a brand strategist for a streetwear company. "
            "You have access to tools that search a visual archive of brand assets "
            "(images, video frames, product photos). Use them to answer questions "
            "about trends, brand alignment, and creative direction.\n\n"
            "Always search before answering — never guess about what's in the archive."
        ),
    )

    questions = [
        "What are the key visual trends in BAPE's archive?",
        "Does BAPE have items that align with minimalist streetwear?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        print("-" * 40)
        result = agent.invoke({"messages": [("human", q)]})
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                print(f"Answer: {msg.content}\n")
                break


if __name__ == "__main__":
    # Demo 0: Raw retriever (always runs — no LLM needed)
    run_retriever_e2e()

    # Demo 1.5: Tool E2E (always runs — no LLM needed)
    run_tool_e2e()

    # Demo 1: RAG chain (needs LLM)
    try:
        run_retriever_chain()
    except Exception as e:
        print(f"\nSkipped — LLM error: {e}\n")

    # Demo 2: Agentic tool use (needs LLM)
    try:
        run_agent()
    except Exception as e:
        print(f"\nSkipped — LLM error: {e}\n")
