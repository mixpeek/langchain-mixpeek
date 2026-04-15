# langchain-mixpeek

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-mixpeek)](https://pypi.org/project/langchain-mixpeek/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-mixpeek)](https://pypistats.org/packages/langchain-mixpeek)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-mixpeek)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/langchain-mixpeek)](https://pypi.org/project/langchain-mixpeek/)
[![Tests](https://img.shields.io/badge/tests-89%20passed-brightgreen)]()

**AI agents are blind. Mixpeek gives them eyes, ears, and memory.**

The official [LangChain](https://langchain.com) integration for [Mixpeek](https://mixpeek.com) — the infrastructure layer that lets AI agents search video, images, audio, and documents through one API.

## Quick install

```bash
pip install langchain-mixpeek
```

## What's inside

| Component | Class | What it does |
|-----------|-------|-------------|
| **Retriever** | `MixpeekRetriever` | Search across video, images, audio, documents — returns LangChain `Document` objects |
| **Tool** | `MixpeekTool` | Standalone agent tool for search — returns JSON for LLM consumption |
| **Toolkit** | `MixpeekToolkit` | 6-tool suite: search, ingest, process, classify, cluster, alert |
| **VectorStore** | `MixpeekVectorStore` | Full CRUD — ingest any file type, search, and manage platform features |
| **Async** | `AsyncMixpeekRetriever` | Non-blocking retriever for async chains and agents |

## Usage

### Retriever

```python
from langchain_mixpeek import MixpeekRetriever

retriever = MixpeekRetriever(
    api_key="mxp_...",
    retriever_id="ret_abc123",
    namespace="my-namespace",
)
docs = retriever.invoke("find the red cup")
```

### One-line agent tool

```python
tool = retriever.as_tool()  # that's it — retriever becomes an agent tool
```

### Agent with 6 capabilities

```python
from langchain_mixpeek import MixpeekToolkit
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

toolkit = MixpeekToolkit(
    api_key="mxp_...",
    namespace="brand-protection",
    bucket_id="bkt_...",
    collection_id="col_...",
    retriever_id="ret_...",
)

agent = create_react_agent(
    ChatAnthropic(model="claude-sonnet-4-20250514"),
    toolkit.get_tools(),
)

result = agent.invoke({
    "messages": [("user", "Scan these product URLs and alert me about counterfeits")]
})
```

The toolkit gives your agent:

| Tool | Capability |
|------|-----------|
| `mixpeek_search` | Search video, images, audio, documents by natural language |
| `mixpeek_ingest` | Upload text, images, video, audio, PDFs, spreadsheets |
| `mixpeek_process` | Trigger feature extraction (embedding, OCR, transcription, face detection) |
| `mixpeek_classify` | Run taxonomy classification on documents |
| `mixpeek_cluster` | Group similar documents (kmeans, dbscan, hdbscan) |
| `mixpeek_alert` | Monitor content with webhook, Slack, or email notifications |

Scope what your agent can do:

```python
toolkit.get_tools(actions=["search"])                         # search only
toolkit.get_tools(actions=["search", "ingest", "process"])    # search + upload
toolkit.get_tools()                                           # all 6 tools
```

### VectorStore — ingest any file type

```python
from langchain_mixpeek import MixpeekVectorStore

store = MixpeekVectorStore(
    api_key="mxp_...",
    namespace="my-namespace",
    bucket_id="bkt_...",
    collection_id="col_...",
    retriever_id="ret_...",
)

# Eyes: images and video
store.add_images(["https://example.com/photo.jpg"])
store.add_videos(["https://example.com/clip.mp4"])

# Ears: audio
store.add_audio(["https://example.com/recording.mp3"])

# Memory: text, PDFs, spreadsheets
store.add_texts(["product description..."])
store.add_pdfs(["https://example.com/doc.pdf"])
store.add_excel(["https://example.com/data.xlsx"])

# Process everything (embedding, OCR, face detection, transcription)
store.trigger_processing()

# Search across all modalities
docs = store.similarity_search("red cup on the table")
```

Convert between interfaces anytime:

```python
retriever = store.as_retriever()
tool = store.as_tool()
toolkit = store.as_toolkit()
```

### RAG chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_mixpeek import MixpeekRetriever

retriever = MixpeekRetriever(api_key="mxp_...", retriever_id="ret_...", namespace="ns")
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

prompt = ChatPromptTemplate.from_template(
    "Answer using this context:\n{context}\n\nQuestion: {question}"
)

chain = {"context": retriever, "question": lambda x: x} | prompt | llm
response = chain.invoke("what happens at 2 minutes?")
```

### Multi-retriever agent

```python
from langchain_mixpeek import MixpeekTool
from langgraph.prebuilt import create_react_agent

video_search = MixpeekTool(
    api_key="mxp_...", retriever_id="ret_video", namespace="archive",
    name="search_videos", description="Search video archive for scenes or faces.",
)
image_search = MixpeekTool(
    api_key="mxp_...", retriever_id="ret_images", namespace="catalog",
    name="search_images", description="Search product images by visual similarity.",
)

agent = create_react_agent(llm, [video_search, image_search])
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | required | Mixpeek API key (`mxp_...`) |
| `retriever_id` | str | required | Retriever ID for search (`ret_...`) |
| `namespace` | str | required | Namespace to operate in |
| `bucket_id` | str | required* | Bucket for uploads (`bkt_...`) |
| `collection_id` | str | required* | Collection for processing (`col_...`) |
| `top_k` | int | `10` / `5` | Max results (retriever / tool) |
| `content_field` | str | `"text"` | Field to use as `page_content` |
| `filters` | dict | `None` | Attribute filters for retriever |

*Required for ingest/processing. Not needed for search-only via `from_retriever()`.

## Platform features

The VectorStore and Toolkit expose the full Mixpeek platform beyond search:

- **Taxonomies** — create and run classification pipelines on your documents
- **Clusters** — group similar content with kmeans, dbscan, hdbscan, or spectral clustering
- **Alerts** — monitor for matches and notify via webhook, Slack, or email
- **Plugins** — manage and test custom feature extractors

## Documentation

- [Full docs](https://docs.mixpeek.com/agent-integrations/langchain) — tutorials, examples, platform features
- [API reference](https://docs.mixpeek.com/integrations/developer-tools/python-sdk) — Python SDK reference
- [Examples](https://github.com/mixpeek/langchain-mixpeek/tree/main/examples) — runnable demo scripts

## License

MIT
