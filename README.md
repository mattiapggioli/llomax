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
│     SAM  (OpenVINO / CPU fallback)      │
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
                       ▼
┌─────────────────────────────────────────┐
│  6. ANNOTATE                            │
│     PlaceholderAnnotator                │
│     label + description per fragment    │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│  7. COMPOSE                             │
│     PIL alpha-composite → RGB canvas    │
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
- A SAM model checkpoint (required for real segmentation; not needed when using the placeholder client)

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

To use real SAM segmentation instead of the placeholder client, download a checkpoint:

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

### Python API

```python
import asyncio
from llomax import Pipeline
from llomax.analysis.client import PlaceholderAnalysisClient  # no checkpoint needed
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

To use SAM segmentation, swap in `Segmenter`:

```python
from llomax.analysis.segmenter import Segmenter

pipeline = Pipeline(
    search_agent=InternetArchiveAgent(),
    analysis_client=Segmenter(
        checkpoint_path="sam_vit_b_01ec64.pth",
        model_type="vit_b",
        device="AUTO",       # AUTO lets OpenVINO pick GPU/NPU; use "CPU" to force CPU
    ),
)
```

## Development

```bash
uv run pytest tests/ -v         # Run all tests
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
```

## Project Structure

```
src/llomax/
├── models.py          # SourceImage, Fragment, CollageOutput
├── output.py          # Save collage and metadata to timestamped directories
├── pipeline.py        # Five-stage pipeline orchestrator
├── search/            # Stage 1 & 2: discovery and source selection
│   ├── internet_archive_agent.py   # InternetArchiveAgent (multi-turn agent loop)
│   ├── clients/
│   │   └── internet_archive_client.py  # InternetArchiveClient
│   ├── curator.py                  # select_sources() via Claude API
│   └── thumbnails.py               # Async thumbnail downloader (saves to disk)
├── analysis/          # Stage 3 & 4: segmentation and annotation
│   ├── segmenter.py   # Segmenter — SAM + OpenVINO (Intel Arc GPU/NPU)
│   ├── annotator.py   # PlaceholderAnnotator — double-analysis structure
│   └── client.py      # AnalysisClient protocol + PlaceholderAnalysisClient
└── composition/       # Stage 5: collage assembly
    └── composer.py    # RGBA alpha-composite composer
```

