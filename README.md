# llomax

LLM agent-based pipeline for creating artistic collages from Internet Archive material.

## Overview

llomax is a three-stage pipeline:

1. **Search** — An LLM agent (Claude) autonomously queries the Internet Archive, issuing multiple searches with varied queries to gather diverse source images.
2. **Analysis** — Visual NER identifies and crops relevant elements (people, objects, text, landmarks) from downloaded images.
3. **Composition** — Cropped elements are arranged into a final collage.

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Copy the example env file and fill in your values
cp .env.example .env
```

Edit `.env` with your settings:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
OUTPUT_DIR=output
```

## Usage

```python
import asyncio
from llomax import Pipeline, SearchResult
from llomax.search.agent import SearchAgent
from llomax.analysis.client import PlaceholderAnalysisClient
from llomax.composition.composer import compose

async def main():
    agent = SearchAgent()
    client = PlaceholderAnalysisClient()
    pipeline = Pipeline(search_agent=agent, analysis_client=client, compose_fn=compose)

    # Collage and metadata are automatically saved to OUTPUT_DIR/{timestamp}/
    collage = await pipeline.run("vintage botanical illustrations", canvas_size=(1920, 1080))

asyncio.run(main())
```

## Development

```bash
uv run pytest tests/ -v         # Run tests
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
```

## Project Structure

```
src/llomax/
├── models.py          # Domain models (SearchResult, AnalysisResult, CollageOutput)
├── output.py          # Save collage and metadata to timestamped directories
├── pipeline.py        # End-to-end pipeline orchestrator
├── search/            # Stage 1: LLM-driven Internet Archive search
│   ├── internet_archive_agent.py  # InternetArchiveAgent with multi-turn agent loop
│   ├── clients/
│   │   └── internet_archive_client.py  # InternetArchiveClient wrapping the internetarchive library
│   ├── curator.py                 # Asset curation via Claude API
│   └── thumbnails.py              # Async thumbnail downloader
├── analysis/          # Stage 2: Visual analysis and cropping
│   └── client.py      # AnalysisClient protocol + placeholder implementation
└── composition/       # Stage 3: Collage assembly
    └── composer.py    # Random placement composer
```

## License

All rights reserved.
