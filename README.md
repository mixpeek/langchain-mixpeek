# langchain-mixpeek

LangChain integration for [Mixpeek](https://mixpeek.com) — multimodal retriever and tool for searching video, image, audio, and document content from AI agents and RAG pipelines.

## Install

```bash
pip install langchain-mixpeek
```

## Quick start

### Retriever (5 lines)

```python
from langchain_mixpeek import MixpeekRetriever

retriever = MixpeekRetriever(
    api_key="mxp_...",
    retriever_id="ret_abc123",
    namespace="my-namespace",
)
docs = retriever.invoke("find the red cup")
```

### Tool (5 lines)

```python
from langchain_mixpeek import MixpeekTool

tool = MixpeekTool(
    api_key="mxp_...",
    retriever_id="ret_abc123",
    namespace="my-namespace",
)
result = tool.invoke("find the red cup")  # returns JSON string
```

### Async retriever

```python
from langchain_mixpeek import AsyncMixpeekRetriever

retriever = AsyncMixpeekRetriever(
    api_key="mxp_...",
    retriever_id="ret_abc123",
    namespace="my-namespace",
)
docs = await retriever.ainvoke("find the red cup")
```

### Use in a chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mixpeek import MixpeekRetriever

retriever = MixpeekRetriever(api_key="mxp_...", retriever_id="ret_...", namespace="ns")
llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_template(
    "Answer using this context:\n{context}\n\nQuestion: {question}"
)

chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
)
response = chain.invoke("what happens at 2 minutes?")
```

### Use as an agent tool

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_mixpeek import MixpeekTool

tools = [MixpeekTool(api_key="mxp_...", retriever_id="ret_...", namespace="ns")]
agent = create_openai_tools_agent(ChatOpenAI(), tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
executor.invoke({"input": "What's in the video at the 2-minute mark?"})
```

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | str | required | Mixpeek API key (`mxp_...`) |
| `retriever_id` | str | required | Mixpeek retriever ID (`ret_...`) |
| `namespace` | str | required | Mixpeek namespace to search |
| `top_k` | int | `10` / `5` | Max results (retriever / tool) |
| `content_field` | str | `"transcript_chunk"` | Metadata field used as `page_content` |
| `filters` | dict | `None` | Attribute filters passed to the retriever |

## Full docs

[docs.mixpeek.com/agent-integrations/langchain](https://docs.mixpeek.com/agent-integrations/langchain)
