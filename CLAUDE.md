# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
uv sync                              # Install dependencies
uv run pytest tests/ -v              # Run all tests (27 tests)
uv run pytest tests/test_search_agent.py -v   # Run a single test file
uv run pytest tests/ -k "test_name"  # Run a specific test by name
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/       # Format
```

Pytest is configured with `asyncio_mode = "auto"` — async test functions run automatically without markers. Ruff line length is 99.

## Architecture

Three-stage pipeline orchestrated by `Pipeline` (`src/llomax/pipeline.py`):

```
SearchAgent.search(prompt) → AnalysisClient.analyze(results) → compose(elements, canvas_size)
```

### Stage 1: Search (`src/llomax/search/`)

`SearchAgent` runs a multi-turn LLM agent loop (max 15 turns) that uses Claude via the Anthropic API to decide which Internet Archive searches to perform. It communicates with the `internet-archive-mcp` server over stdio transport.

- **`agent.py`** — `SearchAgent` class: opens MCP session, runs agent loop, deduplicates results by identifier, downloads thumbnails
- **`mcp.py`** — `open_mcp_session()` context manager (spawns MCP server via `uv run`), `mcp_tools_to_anthropic()` format converter, `forward_tool_calls()` to dispatch LLM tool-use blocks to MCP
- **`parsing.py`** — `parse_search_results()` extracts `SearchResult` objects from MCP tool response JSON
- **`thumbnails.py`** — `download_thumbnails()` async batch downloader using httpx

The MCP server path defaults to `/home/mattiapggl/mcp-servers/internet-archive-mcp` and is configurable via the `SearchAgent` constructor.

### Stage 2: Analysis (`src/llomax/analysis/`)

`AnalysisClient` is a `Protocol` with a single method `analyze(images) -> list[AnalysisResult]`. Currently only `PlaceholderAnalysisClient` exists (passthrough, labels everything `"unknown"`). This is the main extension point — replace with a real NER/vision backend.

### Stage 3: Composition (`src/llomax/composition/`)

`compose()` function places cropped elements onto a white canvas at random positions. Returns `CollageOutput` with the composed PIL Image.

### Domain Models (`src/llomax/models.py`)

- `SearchResult` — Internet Archive item with identifier, title, URLs, optional downloaded `Image`
- `AnalysisResult` — Cropped element with source identifier, entity label, `Image`
- `CollageOutput` — Final composed `Image` with canvas dimensions

All are `@dataclass` types.

## Code Conventions

- Async-first for all I/O (MCP, HTTP). Tests use pytest-asyncio auto mode.
- `from __future__ import annotations` in all modules.
- Protocol-based abstractions for swappable backends (see `AnalysisClient`).
- Each package has explicit `__all__` exports in `__init__.py`.

## Internet Archive MCP Server Reference

The search stage uses the `internet-archive-mcp` MCP server (at `/home/mattiapggl/mcp-servers/internet-archive-mcp`). Three tools over stdio:

| Tool | Parameters | Returns |
|------|-----------|---------|
| `search_images_tool` | `query` (required), `max_results=10`, `collection=None` | JSON array of `{identifier, title, creator, date, description, thumbnail_url, details_url}` |
| `list_curated_collections_tool` | none | JSON array of `{identifier, title, description}` — curated collections (nasa, flickrcommons, smithsonian, etc.) |
| `search_collections_tool` | `query` (required), `max_results=10` | JSON array of `{identifier, title, description, details_url}` |

- `thumbnail_url`: `https://archive.org/services/img/{identifier}` (low-res preview)
- `details_url`: `https://archive.org/details/{identifier}` (item page)
- Full-resolution download: `https://archive.org/download/{identifier}/{filename}` or use the `internetarchive` Python library

## Environment

Requires `ANTHROPIC_API_KEY` to be set for the search agent's LLM calls.
