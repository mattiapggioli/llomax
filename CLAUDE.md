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

Eight-stage pipeline orchestrated by `Pipeline` (`src/llomax/pipeline.py`):

```
InternetArchiveAgent.plan_search(prompt)   # LLM registers search intents (no execution)
  → _execute_search_plan(plan)             # Python runs each query via InternetArchiveClient
  → download_thumbnails(sources)           # save to cache_dir/{id}.jpg
  → AnalysisClient.analyze(sources)        # segmentation → Fragments
  → select_fragments(fragments)            # LLM curator picks individual Fragments
  ── after_curation hooks ──
  → PlaceholderAnnotator.annotate()        # label + describe each Fragment
  ── pre_composition hooks ──
  → compose(fragments, canvas_size)        # RGBA alpha-composite onto canvas
                                           # (or composition_strategy override)
  → save_run(collage, ...)                 # write collage.png + metadata.json
```

### Hook architecture (`src/llomax/core/hooks.py`, `src/llomax/hooks/`)

`HookManager` manages two kinds of extensions:
- **Additive hooks** (`register`) — async callables that receive and mutate `PipelineState`. Multiple hooks on the same point run in registration order.
- **Override hooks** (`register_override`) — replace the pipeline's default at that point. Last registration wins. Used for `composition_strategy`.

`PipelineState` fields: `prompt`, `canvas_size`, `sources`, `fragments`, `background_source_id`, `background_image`.

Three built-in hook factories in `src/llomax/hooks/`:
- **`select_best_background(client)`** (`background.py`) — `after_curation`. LLM picks one source as canvas background; sets `state.background_source_id`. Text-only, no Vision API.
- **`color_grade(mode)`** (`palette.py`) — `pre_composition`. Applies pastel/vivid/vintage/faded PIL transforms to all fragments and background. Preserves RGBA alpha.
- **`llm_compose(client)`** (`llm_composer.py`) — `composition_strategy` override. LLM returns `{fragment_id: {x, y, scale, reason}}`; falls back to random on failure.

All built-in hooks use `claude-haiku-4-5-20251001`.

### Stage 1: Discovery (`src/llomax/search/`)

`InternetArchiveAgent` (`internet_archive_agent.py`) runs a structured LLM agent loop (max 10 turns) with two tools that dispatch to `InternetArchiveClient`:

- **`search_images`** — accepts `keywords: list[str]` joined with OR by default. Returns `{"results": [...], "count": N}`; adds a `"suggestion"` key when `count == 0` to guide the agent toward a semantic fallback. Mediatype:image is enforced by the client.
- **`find_collections`** — discovers IA collections by keyword list (OR-joined). Mediatype:collection is enforced.

`plan_search()` records search intents without executing them. `_execute_search_plan()` in `Pipeline` runs them directly via `InternetArchiveClient`. The planner targets a candidate pool of **5× max_items**.

Supporting files:
- **`clients/internet_archive_client.py`** — `InternetArchiveClient`. `_build_query` joins a `list[str]` with OR and strips terms implicit to the collection via `_COLLECTION_IMPLICIT_TERMS` (e.g. "space" is redundant when `collection="nasa"`). Falls back to the original keyword list if all terms would be stripped.
- **`thumbnails.py`** — `download_thumbnails(sources, cache_dir)` saves each thumbnail as `{cache_dir}/{external_id}.jpg`. Already-cached files are reused.

### Stage 2: Curator (`src/llomax/search/curator.py`)

`select_fragments()` takes all extracted `Fragment` objects (label, pixel dimensions, parent source context) and makes a single `claude-haiku-4-5-20251001` call to pick the best subset for the collage. Returns a list of selected `fragment_id` strings. No tool use — structured JSON output only.

### Stage 3: Segmentation (`src/llomax/analysis/`)

`AnalysisClient` is a `Protocol` with a single async method `analyze(sources) -> list[Fragment]`.

- **`Segmenter`** (`segmenter.py`) — SAM `AutomaticMaskGenerator` backend. On first use it exports the SAM image encoder (ViT) to OpenVINO IR and compiles it with `device_name="AUTO"` to target the Intel Arc GPU or NPU. Falls back to PyTorch CPU if OpenVINO is unavailable. Each mask becomes a `Fragment` with a transparent RGBA background.
- **`YoloAnalysisClient`** (`yolo_client.py`) — ultralytics `-seg` model backend. Runs instance segmentation; each detected instance becomes a `Fragment` with a masked RGBA crop.
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
- `Fragment` — Extracted visual segment: `source_id`, `image_rgba` (RGBA PIL Image), `bounding_box` (x1,y1,x2,y2), `label`, `description`, `fragment_id` (auto-generated UUID).
- `CollageOutput` — Final composed RGB `Image` with canvas dimensions and `fragment_provenance` list.

All are `@dataclass` types.

## Code Conventions

- Async-first for all I/O (HTTP, Anthropic API). Tests use pytest-asyncio auto mode.
- `from __future__ import annotations` in all modules.
- All logging via `loguru.logger`. Pipeline adds a per-run file sink at `{run_dir}/pipeline.log`.
- Protocol-based abstractions for swappable backends (see `AnalysisClient`).
- Each package has explicit `__all__` exports in `__init__.py`.
- No module-level docstrings. Classes, methods, and functions use Google-style docstrings.

## Environment

Configuration is loaded from a `.env` file at the project root (via python-dotenv). Copy `.env.example` to `.env` and fill in values. The `.env` file is gitignored.

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | API key for the search and curator agent LLM calls |
| `OUTPUT_DIR` | No | Base directory for pipeline run outputs (default: `output`) |
