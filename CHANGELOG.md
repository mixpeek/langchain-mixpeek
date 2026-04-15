# Changelog

## 0.4.2 (2026-04-15)

- Rewrite README with badges, component table, and full usage examples
- Fix `content_field` default in docs (was `"transcript_chunk"`, now correctly `"text"`)
- Update PyPI description and keywords

## 0.4.0 (2026-04-15)

- Add `MixpeekToolkit` — 6-tool agent suite (search, ingest, process, classify, cluster, alert)
- Add `MixpeekVectorStore` with full platform features (taxonomies, clusters, alerts, plugins)
- Add `as_tool()` bridge on `MixpeekRetriever` — one-line retriever-to-tool conversion
- Add `as_retriever()`, `as_toolkit()`, `from_retriever()` bridge methods on VectorStore
- Add multimodal ingestion: `add_images()`, `add_videos()`, `add_audio()`, `add_pdfs()`, `add_excel()`
- 89 tests covering all components

## 0.2.0 (2026-04-14)

- Add `MixpeekTool` for agent integration
- Add `AsyncMixpeekRetriever` for async chains
- Add attribute filtering via `filters` parameter

## 0.1.0 (2026-04-14)

- Initial release
- `MixpeekRetriever` — LangChain BaseRetriever for multimodal search
- Support for LCEL chains and RAG pipelines
