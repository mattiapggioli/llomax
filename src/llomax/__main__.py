from __future__ import annotations

import argparse
import asyncio

from llomax.analysis.client import PlaceholderAnalysisClient
from llomax.pipeline import Pipeline
from llomax.search.internet_archive_agent import InternetArchiveAgent


def _parse_canvas(value: str) -> tuple[int, int]:
    """Parse a 'WIDTHxHEIGHT' string into a (width, height) tuple."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"canvas must be WIDTHxHEIGHT, got '{value}'")
    return int(parts[0]), int(parts[1])


async def _run(prompt: str, canvas_size: tuple[int, int], max_items: int) -> None:
    """Run the pipeline with the given prompt and canvas size.

    Uses ``PlaceholderAnalysisClient`` by default (no SAM checkpoint required).
    To use SAM segmentation, replace it with ``Segmenter(checkpoint_path=...)``.

    Args:
        prompt: Creative text prompt describing the desired collage.
        canvas_size: ``(width, height)`` in pixels.
        max_items: Target number of source images for curation.
    """
    agent = InternetArchiveAgent()
    client = PlaceholderAnalysisClient()
    pipeline = Pipeline(search_agent=agent, analysis_client=client)
    await pipeline.run(prompt, canvas_size=canvas_size, max_items=max_items)


def cli() -> None:
    """Entry point for the ``llomax`` console script."""
    parser = argparse.ArgumentParser(
        description="Generate an artistic collage from Internet Archive material.",
    )
    parser.add_argument("prompt", help="Creative text prompt describing the desired collage")
    parser.add_argument(
        "--canvas",
        type=_parse_canvas,
        default="1024x1024",
        help="Canvas size as WIDTHxHEIGHT (default: 1024x1024)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="Target number of source images to curate (default: 20)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.prompt, args.canvas, args.max_items))


if __name__ == "__main__":
    cli()
