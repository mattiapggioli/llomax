# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Writing Guidelines

- Describe what something *is*, not what it *isn't*. Don't negate what's absent — state what's present.

## Development Commands

```bash
uv sync                              # Install dependencies
uv run pytest tests/ -v              # Run all tests (37 tests)
uv run pytest tests/test_search_agent.py -v   # Run a single test file
uv run pytest tests/ -k "test_name"  # Run a specific test by name
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/       # Format
```

Pytest is configured with `asyncio_mode = "auto"` — async test functions run automatically without markers. Ruff line length is 99.

## Architecture

Four-stage pipeline orchestrated by `Pipeline` (`src/llomax/pipeline.py`):

```
InternetArchiveAgent.search(prompt) → Curation (select_assets) → AnalysisClient.analyze(results) → compose(elements, canvas_size)
```

### Stage 1: Discovery (`src/llomax/search/`)

`InternetArchiveAgent` (`internet_archive_agent.py`) runs a structured LLM agent loop (max 10 turns) with two "blinded" tools that dispatch directly to `IAClient`:

- **`search_images`** — Lucene boolean keywords, optional collection/date filter. Mediatype:image is enforced by the client.
- **`find_collections`** — Discover IA collections by keyword. Mediatype:collection is enforced.

Supporting files:
- **`clients/ia_client.py`** — `IAClient` wraps the `internetarchive` Python library. Three methods: `search_images()`, `find_collections()`, `get_curated_collections()`. All enforce correct mediatypes.
- **`thumbnails.py`** — `download_thumbnails()` async batch downloader using httpx

### Stage 2: Curation (`src/llomax/search/curator.py`)

`select_assets()` takes sanitized candidates (identifier, title, description, year) and makes a single Claude API call (claude-sonnet-4-5-20250929) to select the best assets for the collage. Returns a JSON array of selected identifiers. No tool use — structured output only.

### Stage 3: Analysis (`src/llomax/analysis/`)

`AnalysisClient` is a `Protocol` with a single method `analyze(images) -> list[AnalysisResult]`. Currently only `PlaceholderAnalysisClient` exists (passthrough, labels everything `"unknown"`). This is the main extension point — replace with a real NER/vision backend.

### Stage 4: Composition (`src/llomax/composition/`)

`compose()` function places cropped elements onto a white canvas at random positions. Returns `CollageOutput` with the composed PIL Image.

### Domain Models (`src/llomax/models.py`)

- `SearchResult` — Internet Archive item with identifier, title, URLs, description, year, optional downloaded `Image`
- `AnalysisResult` — Cropped element with source identifier, entity label, `Image`
- `CollageOutput` — Final composed `Image` with canvas dimensions

All are `@dataclass` types.

## Code Conventions

- Async-first for all I/O (HTTP, Anthropic API). Tests use pytest-asyncio auto mode.
- `from __future__ import annotations` in all modules.
- Protocol-based abstractions for swappable backends (see `AnalysisClient`).
- Each package has explicit `__all__` exports in `__init__.py`.

## Internet Archive Client Reference

The search stage uses the `internetarchive` Python library directly via `IAClient` (`src/llomax/search/clients/ia_client.py`). Three methods:

| Method | Parameters | Returns |
|--------|-----------|---------|
| `search_images` | `keywords` (required), `collection=None`, `date_filter=None`, `max_results=20` | `list[ImageResult]` — `{identifier, title, creator, date, description, thumbnail_url, details_url}` |
| `find_collections` | `keywords` (required), `max_results=10` | `list[CollectionResult]` — `{identifier, title, description, details_url}` |
| `get_curated_collections` | none | `list[CuratedCollection]` — curated collections (nasa, flickrcommons, smithsonian, etc.) |

- `thumbnail_url`: `https://archive.org/services/img/{identifier}` (low-res preview)
- `details_url`: `https://archive.org/details/{identifier}` (item page)
- Full-resolution download: `https://archive.org/download/{identifier}/{filename}`

## Environment

Configuration is loaded from a `.env` file at the project root (via python-dotenv). Copy `.env.example` to `.env` and fill in values. The `.env` file is gitignored.

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | API key for the search and curator agent LLM calls |
| `OUTPUT_DIR` | No | Base directory for pipeline run outputs (default: `output`) |
