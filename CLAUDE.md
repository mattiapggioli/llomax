# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Writing Guidelines

- Describe what something *is*, not what it *isn't*. Don't negate what's absent — state what's present.

## Development Commands

```bash
uv sync                              # Install dependencies
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/test_search_agent.py -v   # Run a single test file
uv run pytest tests/ -k "test_name"  # Run a specific test by name
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/       # Format
```

Pytest is configured with `asyncio_mode = "auto"` — async test functions run automatically without markers. Ruff line length is 99.

## Architecture

Five-stage pipeline orchestrated by `Pipeline` (`src/llomax/pipeline.py`):

```
InternetArchiveAgent.plan_search(prompt)
  → select_sources(candidates)          # LLM curator picks SourceImages
  → download_thumbnails(sources)        # save to cache_dir/{id}.jpg
  → AnalysisClient.analyze(sources)     # SAM segmentation → Fragments
  → PlaceholderAnnotator.annotate()     # label + describe each Fragment
  → compose(fragments, canvas_size)     # RGBA alpha-composite onto canvas
```

### Stage 1: Discovery (`src/llomax/search/`)

`InternetArchiveAgent` (`internet_archive_agent.py`) runs a structured LLM agent loop (max 10 turns) with two "blinded" tools that dispatch directly to `InternetArchiveClient`:

- **`search_images`** — Lucene boolean keywords, optional collection/date filter. Mediatype:image is enforced by the client.
- **`find_collections`** — Discover IA collections by keyword. Mediatype:collection is enforced.

`plan_search()` records search intents without executing them; `_execute_search_plan()` in `Pipeline` runs them in Python.

Supporting files:
- **`clients/internet_archive_client.py`** — `InternetArchiveClient` wraps the `internetarchive` Python library. Three methods: `search_images()`, `find_collections()`, `get_curated_collections()`. All enforce correct mediatypes.
- **`thumbnails.py`** — `download_thumbnails(sources, cache_dir)` saves each thumbnail as `{cache_dir}/{external_id}.jpg` and sets `source.local_path`. Already-cached files are reused.

### Stage 2: Source Selection (`src/llomax/search/curator.py`)

`select_sources()` takes `SourceImage` candidates (external_id, title, description, year, creator) and makes a single Claude API call (claude-sonnet-4-5-20250929) to select the best source images for the collage. Returns a JSON array of selected `external_id` strings. No tool use — structured output only.

### Stage 3: Segmentation (`src/llomax/analysis/`)

`AnalysisClient` is a `Protocol` with a single async method `analyze(sources) -> list[Fragment]`.

- **`Segmenter`** (`segmenter.py`) — SAM `AutomaticMaskGenerator` backend. On first use it exports the SAM image encoder (ViT) to OpenVINO IR format and compiles it with `device_name="AUTO"` to target the Intel Arc GPU or NPU. Falls back to PyTorch CPU if OpenVINO is unavailable. Each mask becomes a `Fragment` with a transparent RGBA background. Also implements `analyze()` for protocol conformance.
- **`PlaceholderAnalysisClient`** (`client.py`) — Passthrough: wraps each source image as a single full-frame `Fragment`. No model required; use for testing the pipeline end-to-end.

### Stage 4: Annotation (`src/llomax/analysis/annotator.py`)

`PlaceholderAnnotator` implements the double-analysis structure:
- `annotate_source(source)` — high-level context string for the full IA item.
- `annotate_fragment(fragment, source)` — region-level description referencing parent metadata.
- `annotate(sources, fragments)` — populates `fragment.description` in place for all fragments.

Replace method bodies with live `anthropic.AsyncAnthropic` vision calls once a real backend is ready.

### Stage 5: Composition (`src/llomax/composition/`)

`compose(fragments, canvas_size, background)` places each `Fragment.image_rgba` at a random position using `PIL.Image.paste` with the alpha channel as a mask. Returns `CollageOutput` with the final RGB image.

### Domain Models (`src/llomax/models.py`)

- `SourceImage` — Internet Archive item: `external_id`, `title`, `description`, `local_path`, `metadata` dict (creator, year, thumbnail_url, details_url). `load_image()` reads from `local_path`.
- `Fragment` — Extracted visual segment: `source_id`, `image_rgba` (RGBA PIL Image), `bounding_box` (x1,y1,x2,y2), `label`, `description`.
- `CollageOutput` — Final composed RGB `Image` with canvas dimensions and `fragment_provenance` list.

All are `@dataclass` types.

## Code Conventions

- Async-first for all I/O (HTTP, Anthropic API). Tests use pytest-asyncio auto mode.
- `from __future__ import annotations` in all modules.
- Protocol-based abstractions for swappable backends (see `AnalysisClient`).
- Each package has explicit `__all__` exports in `__init__.py`.
- No module-level docstrings. Classes, methods, and functions use Google-style docstrings.

## Environment

Configuration is loaded from a `.env` file at the project root (via python-dotenv). Copy `.env.example` to `.env` and fill in values. The `.env` file is gitignored.

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | API key for the search and curator agent LLM calls |
| `OUTPUT_DIR` | No | Base directory for pipeline run outputs (default: `output`) |
