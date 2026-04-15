"""
Agents Are Blind — Mixpeek Gives Them Eyes

This demo shows LangChain agents using Mixpeek to search, ingest, classify,
cluster, and monitor multimodal content (images, video, audio, documents).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...  # or OPENAI_API_KEY
    python examples/agent_demo.py

Requires:
    pip install langchain-mixpeek langchain-anthropic langgraph
"""

import os
import sys

from langchain_core.prompts import ChatPromptTemplate

from langchain_mixpeek import MixpeekRetriever, MixpeekTool, MixpeekToolkit, MixpeekVectorStore

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


# ---------------------------------------------------------------------------
# Demo 0: Raw retriever (no LLM)
# ---------------------------------------------------------------------------

def run_retriever_e2e():
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


# ---------------------------------------------------------------------------
# Demo 1: as_tool() — retriever becomes an agent tool in one line
# ---------------------------------------------------------------------------

def run_as_tool():
    print("=" * 60)
    print("DEMO 1: retriever.as_tool() — one-line conversion")
    print("=" * 60)

    import json

    retriever = MixpeekRetriever(
        api_key=MIXPEEK_API_KEY,
        retriever_id=RETRIEVER_ID,
        namespace=NAMESPACE,
        top_k=3,
        content_field="trend_insight",
    )

    tool = retriever.as_tool()
    print(f"\n  Tool name: {tool.name}")
    print(f"  Tool description: {tool.description[:80]}...")

    result = tool.invoke("bold graphics")
    parsed = json.loads(result)
    print(f"\n  Results: {len(parsed)} documents")
    for r in parsed[:2]:
        print(f"    score={r['score']} | {r['content'][:120]}")
    print()


# ---------------------------------------------------------------------------
# Demo 2: RAG chain
# ---------------------------------------------------------------------------

def run_retriever_chain():
    print("=" * 60)
    print("DEMO 2: Retriever → RAG Chain")
    print("=" * 60)

    llm = _get_llm()
    if llm is None:
        print("\nSkipped — no LLM API key (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        return

    retriever = MixpeekRetriever(
        api_key=MIXPEEK_API_KEY,
        retriever_id=RETRIEVER_ID,
        namespace=NAMESPACE,
        top_k=5,
        content_field="trend_insight",
    )

    prompt = ChatPromptTemplate.from_template(
        "You are a brand analyst. Use the search results below to answer.\n\n"
        "Search results:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely, citing specific items."
    )

    chain = {"context": retriever, "question": lambda x: x} | prompt | llm

    question = "What streetwear trends connect to BAPE's archive?"
    print(f"\nQuestion: {question}")
    print("-" * 40)
    response = chain.invoke(question)
    print(f"Answer: {response.content}\n")


# ---------------------------------------------------------------------------
# Demo 3: ReAct agent with multi-retriever tools
# ---------------------------------------------------------------------------

def run_agent():
    print("=" * 60)
    print("DEMO 3: ReAct Agent with Multiple MixpeekTools")
    print("=" * 60)

    llm = _get_llm()
    if llm is None:
        print("\nSkipped — no LLM API key (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")
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
            "Input: natural language query about fashion trends or aesthetics."
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
            "Check whether items in the archive align with a style or concept. "
            "Input: description of a style to check against."
        ),
        content_field="brand_alignment",
    )

    agent = create_react_agent(
        llm,
        [trend_tool, brand_tool],
        prompt=(
            "You are a brand strategist for a streetwear company. "
            "Use tools to search the visual archive before answering."
        ),
    )

    question = "What are the key visual trends in BAPE's archive?"
    print(f"\nQuestion: {question}")
    print("-" * 40)
    result = agent.invoke({"messages": [("human", question)]})
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            print(f"Answer: {msg.content}\n")
            break


# ---------------------------------------------------------------------------
# Demo 4: MixpeekToolkit — full agent with 6 capabilities
# ---------------------------------------------------------------------------

def run_toolkit_agent():
    print("=" * 60)
    print("DEMO 4: MixpeekToolkit — Full Agent (search + ingest + classify)")
    print("=" * 60)

    llm = _get_llm()
    if llm is None:
        print("\nSkipped — no LLM API key (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        return

    from langgraph.prebuilt import create_react_agent

    toolkit = MixpeekToolkit(
        api_key=MIXPEEK_API_KEY,
        namespace=NAMESPACE,
        bucket_id="bkt_0184971a",
        collection_id="col_ae05ab395a",
        retriever_id=RETRIEVER_ID,
    )

    print(f"\n  Tools available: {[t.name for t in toolkit.get_tools()]}")

    # Give the agent search + ingest + process (scoped)
    agent = create_react_agent(
        llm,
        toolkit.get_tools(actions=["search", "ingest", "process"]),
        prompt=(
            "You are a multimodal AI assistant with eyes and ears. "
            "You can search video, images, and audio content using mixpeek_search. "
            "You can upload new content using mixpeek_ingest. "
            "You can trigger processing using mixpeek_process. "
            "Always search before answering questions about content."
        ),
    )

    question = "Search the archive for camo patterns and describe what you find."
    print(f"\n  Question: {question}")
    print("-" * 40)
    result = agent.invoke({"messages": [("human", question)]})
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            print(f"  Answer: {msg.content[:500]}\n")
            break


# ---------------------------------------------------------------------------
# Demo 5: VectorStore bridge methods
# ---------------------------------------------------------------------------

def run_store_bridges():
    print("=" * 60)
    print("DEMO 5: VectorStore → Retriever / Tool / Toolkit bridges")
    print("=" * 60)

    # Minimal config (search-only)
    store = MixpeekVectorStore.from_retriever(
        api_key=MIXPEEK_API_KEY,
        namespace=NAMESPACE,
        retriever_id=RETRIEVER_ID,
        content_field="trend_insight",
    )
    print(f"\n  from_retriever() → store with retriever_id={store.retriever_id}")

    # Convert to retriever
    retriever = store.as_retriever()
    print(f"  as_retriever() → {type(retriever).__name__}")

    # Convert to tool
    tool = store.as_tool()
    print(f"  as_tool() → {tool.name}")

    # Convert to toolkit
    toolkit = store.as_toolkit()
    tools = toolkit.get_tools()
    print(f"  as_toolkit() → {len(tools)} tools: {[t.name for t in tools]}")

    # Actually search
    docs = store.similarity_search("camo", k=2)
    print(f"\n  similarity_search('camo') → {len(docs)} docs")
    if docs:
        print(f"    [{1}] score={docs[0].metadata.get('score'):.3f} | {docs[0].page_content[:100]}")
    print()


if __name__ == "__main__":
    # Always runs (no LLM needed)
    run_retriever_e2e()
    run_as_tool()
    run_store_bridges()

    # Needs LLM
    try:
        run_retriever_chain()
    except Exception as e:
        print(f"\nSkipped — error: {e}\n")

    try:
        run_agent()
    except Exception as e:
        print(f"\nSkipped — error: {e}\n")

    try:
        run_toolkit_agent()
    except Exception as e:
        print(f"\nSkipped — error: {e}\n")
