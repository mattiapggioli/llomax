# llomax

LLM agent-based pipeline for creating artistic collages from Internet Archive material.

## Overview

llomax is an LLM-driven pipeline that turns a text prompt into an artistic collage sourced entirely from the Internet Archive.

```
   prompt
     │
     ▼
┌─────────────────────────────────────────┐
│  1. PLAN SEARCH                         │
│     InternetArchiveAgent  (LLM loop)    │
│     registers search intents            │
└──────────────────────┬──────────────────┘
                       │ search plan
                       ▼
┌─────────────────────────────────────────┐
│  2. EXECUTE PLAN                        │
│     InternetArchiveClient               │
│     OR-joined Lucene queries → ~5×pool  │
└──────────────────────┬──────────────────┘
                       │ SourceImage candidates
                       ▼
┌─────────────────────────────────────────┐
│  3. DOWNLOAD THUMBNAILS                 │
│     async HTTP → cache_dir/{id}.jpg     │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│  4. SEGMENT                             │
│     SAM (OpenVINO / CPU) or YOLO-seg    │
│     one RGBA Fragment per mask          │
└──────────────────────┬──────────────────┘
                       │ Fragment pool
                       ▼
┌─────────────────────────────────────────┐
│  5. CURATE FRAGMENTS                    │
│     LLM single call (Haiku)             │
│     selects fragment_id subset          │
└──────────────────────┬──────────────────┘
                       │ selected Fragments
                ┌──────┴──────┐
                │ after_curation hook point
                └──────┬──────┘
                       ▼
┌─────────────────────────────────────────┐
│  6. ANNOTATE                            │
│     PlaceholderAnnotator                │
│     label + description per fragment    │
└──────────────────────┬──────────────────┘
                       │
                ┌──────┴──────┐
                │ pre_composition hook point
                └──────┬──────┘
                       ▼
┌─────────────────────────────────────────┐
│  7. COMPOSE                             │
│     PIL alpha-composite → RGB canvas    │
│     (or composition_strategy override)  │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│  8. SAVE                                │
│     collage.png  +  metadata.json       │
└─────────────────────────────────────────┘
```

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- An [Anthropic API key](https://console.anthropic.com/)
- A SAM model checkpoint (required for SAM segmentation; not needed for the placeholder or YOLO clients)

## Setup

```bash
# Clone and install dependencies
git clone <repo-url>
cd llomax
uv sync

# Copy the example env file and fill in your Anthropic API key
cp .env.example .env
```

Edit `.env`:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
OUTPUT_DIR=output          # optional, defaults to ./output
```

### SAM model checkpoint (optional)

To use SAM segmentation, download a checkpoint:

```bash
# ViT-B (fastest, ~375 MB) — recommended for the Intel Core Ultra 7 155H
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L (~1.2 GB) or ViT-H (~2.6 GB) for higher mask quality
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

On first run, `Segmenter` automatically exports the encoder to OpenVINO IR and caches it next to the checkpoint in an `ov_cache/` directory.

## Running

### Command line

```bash
uv run llomax "vintage botanical illustrations"
```

```bash
# Custom canvas size and number of source images
uv run llomax "1920s jazz posters" --canvas 1920x1080 --max-items 15

# Quiet run with a specific output directory
OUTPUT_DIR=/tmp/collages uv run llomax "Soviet constructivist propaganda"
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--canvas WxH` | `1024x1024` | Output canvas size in pixels |
| `--max-items N` | `20` | Number of source images to curate |

Output is written to `OUTPUT_DIR/{YYYY-MM-DD_HH-MM-SS}/`:
- `collage.png` — the composed image
- `metadata.json` — prompt, sources, per-fragment provenance
- `pipeline.log` — full debug log of the run

### Python API

```python
import asyncio
from llomax import Pipeline
from llomax.analysis.client import PlaceholderAnalysisClient
from llomax.search.internet_archive_agent import InternetArchiveAgent

async def main():
    pipeline = Pipeline(
        search_agent=InternetArchiveAgent(),
        analysis_client=PlaceholderAnalysisClient(),
    )
    collage = await pipeline.run(
        "vintage botanical illustrations",
        canvas_size=(1920, 1080),
        max_items=15,
    )
    collage.image.show()

asyncio.run(main())
```

**Analysis client options:**

```python
# SAM segmentation (requires checkpoint)
from llomax.analysis.segmenter import Segmenter

analysis_client = Segmenter(
    checkpoint_path="sam_vit_b_01ec64.pth",
    model_type="vit_b",
    device="AUTO",   # AUTO lets OpenVINO pick GPU/NPU; use "CPU" to force CPU
)

# YOLO instance segmentation (requires ultralytics and a -seg model)
from llomax.analysis.yolo_client import YoloAnalysisClient

analysis_client = YoloAnalysisClient(model_name="yolo11n-seg.pt")

# Placeholder — no model required, full-frame fragment per source
from llomax.analysis.client import PlaceholderAnalysisClient

analysis_client = PlaceholderAnalysisClient()
```

## Customizing the Pipeline

The pipeline exposes three extension points via `HookManager`. Hooks receive a `PipelineState` object with:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `str` | Original user prompt |
| `canvas_size` | `tuple[int, int]` | Canvas dimensions in pixels |
| `sources` | `list[SourceImage]` | Curated source images |
| `fragments` | `list[Fragment]` | Selected fragments |
| `background_source_id` | `str \| None` | Source flagged as background (set by hooks) |
| `background_image` | `Image \| None` | Loaded background PIL image (set by pipeline) |

**Hook points:**

| Point | Runs | Type |
|-------|------|------|
| `after_curation` | After fragment curation, before annotation | Additive — mutates `PipelineState` |
| `pre_composition` | After annotation, before composition | Additive — mutates `PipelineState` |
| `composition_strategy` | Replaces the default random composer | Override — returns `CollageOutput` |

### Built-in hooks

| Hook | Import | Hook point | Effect |
|------|--------|------------|--------|
| `select_best_background(client)` | `llomax.hooks` | `after_curation` | LLM picks one source as the canvas background |
| `color_grade(mode)` | `llomax.hooks` | `pre_composition` | Applies a colour style to all fragments and the background |
| `llm_compose(client)` | `llomax.hooks` | `composition_strategy` | LLM places each fragment with x/y/scale and artistic reasoning |

`color_grade` modes: `"pastel"` · `"vivid"` · `"vintage"` · `"faded"`

### Example: full hook setup

```python
import asyncio
import anthropic
from llomax import Pipeline
from llomax.core.hooks import HookManager
from llomax.hooks import color_grade, llm_compose, select_best_background
from llomax.search.internet_archive_agent import InternetArchiveAgent
from llomax.analysis.client import PlaceholderAnalysisClient

async def main():
    client = anthropic.AsyncAnthropic()
    hooks = HookManager()

    # Pick an atmospheric image from the curated pool as canvas background
    hooks.register("after_curation", select_best_background(client))

    # Shift all fragments and background to a unified sepia palette
    hooks.register("pre_composition", color_grade("vintage"))

    # Let Claude place fragments instead of random placement
    hooks.register_override("composition_strategy", llm_compose(client))

    pipeline = Pipeline(
        search_agent=InternetArchiveAgent(),
        analysis_client=PlaceholderAnalysisClient(),
        hooks=hooks,
    )
    collage = await pipeline.run("Soviet constructivist propaganda")
    collage.image.show()

asyncio.run(main())
```

### Writing a custom hook

Additive hooks receive and mutate `PipelineState`:

```python
from llomax.core.hooks import PipelineState

async def drop_small_fragments(state: PipelineState) -> None:
    state.fragments = [
        f for f in state.fragments
        if (f.bounding_box[2] - f.bounding_box[0]) > 100
    ]

hooks.register("after_curation", drop_small_fragments)
```

Override hooks replace the composition stage entirely and return a `CollageOutput`:

```python
from llomax.models import CollageOutput
from PIL import Image

async def tiled_compose(state: PipelineState) -> CollageOutput:
    canvas_w, canvas_h = state.canvas_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    # ... custom tiling logic ...
    return CollageOutput(image=canvas, width=canvas_w, height=canvas_h, fragment_provenance=[])

hooks.register_override("composition_strategy", tiled_compose)
```

Multiple additive hooks on the same point run in registration order. Only the last `register_override` call for a point takes effect.

## Development

```bash
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/test_hooks.py -v # Run a single test file
uv run pytest tests/ -k "test_name"  # Run a specific test by name
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/       # Format
```

## Project Structure

```
src/llomax/
├── models.py               # SourceImage, Fragment, CollageOutput
├── output.py               # Save collage and metadata to timestamped directories
├── pipeline.py             # Pipeline orchestrator (8 stages + 3 hook points)
├── core/
│   └── hooks.py            # HookManager, PipelineState
├── hooks/                  # Built-in hook implementations
│   ├── background.py       # select_best_background
│   ├── palette.py          # color_grade
│   └── llm_composer.py     # llm_compose
├── search/                 # Stages 1–2: discovery and curation
│   ├── internet_archive_agent.py   # InternetArchiveAgent (multi-turn LLM loop)
│   ├── clients/
│   │   └── internet_archive_client.py  # InternetArchiveClient (Lucene queries)
│   ├── curator.py          # select_fragments() — LLM fragment picker
│   └── thumbnails.py       # Async thumbnail downloader
├── analysis/               # Stages 4–5: segmentation and annotation
│   ├── segmenter.py        # Segmenter — SAM + OpenVINO (Intel Arc GPU/NPU)
│   ├── yolo_client.py      # YoloAnalysisClient — YOLO instance segmentation
│   ├── annotator.py        # PlaceholderAnnotator
│   └── client.py           # AnalysisClient protocol + PlaceholderAnalysisClient
└── composition/            # Stage 7: default random composer
    └── composer.py         # RGBA alpha-composite, random placement
```
