from __future__ import annotations

import argparse
import asyncio

from llomax.analysis.client import PlaceholderAnalysisClient, YoloAnalysisClient
from llomax.pipeline import Pipeline
from llomax.search.internet_archive_agent import InternetArchiveAgent


def _parse_canvas(value: str) -> tuple[int, int]:
    """Parse a 'WIDTHxHEIGHT' string into a (width, height) tuple."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"canvas must be WIDTHxHEIGHT, got '{value}'")
    return int(parts[0]), int(parts[1])


async def _run(
    prompt: str,
    canvas_size: tuple[int, int],
    max_items: int,
    segmenter: str,
    yolo_model: str,
) -> None:
    """Run the pipeline with the given prompt and canvas size.

    Args:
        prompt: Creative text prompt describing the desired collage.
        canvas_size: ``(width, height)`` in pixels.
        max_items: Target number of source images to curate.
        segmenter: Analysis backend — ``"yolo"`` or ``"placeholder"``.
        yolo_model: Ultralytics model name used when segmenter is ``"yolo"``.
    """
    agent = InternetArchiveAgent()

    if segmenter == "placeholder":
        analysis_client = PlaceholderAnalysisClient()
    else:
        analysis_client = YoloAnalysisClient(model_name=yolo_model)

    pipeline = Pipeline(search_agent=agent, analysis_client=analysis_client)
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
    parser.add_argument(
        "--segmenter",
        choices=["yolo", "placeholder"],
        default="yolo",
        help="Segmentation backend (default: yolo). Use 'placeholder' for fast testing.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n-seg.pt",
        help="Ultralytics segmentation model name or path (default: yolov8n-seg.pt)",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.prompt, args.canvas, args.max_items, args.segmenter, args.yolo_model))


if __name__ == "__main__":
    cli()
