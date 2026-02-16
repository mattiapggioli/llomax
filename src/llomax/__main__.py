"""Run the llomax pipeline from the command line.

Usage:
    uv run python -m llomax "vintage botanical illustrations"
    uv run python -m llomax "nasa space photos" --canvas 1920x1080
"""

from __future__ import annotations

import argparse
import asyncio

from llomax.analysis.client import PlaceholderAnalysisClient
from llomax.pipeline import Pipeline
from llomax.search.agent import SearchAgent


def parse_canvas(value: str) -> tuple[int, int]:
    """Parse a 'WIDTHxHEIGHT' string into a (width, height) tuple."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"canvas must be WIDTHxHEIGHT, got '{value}'")
    return int(parts[0]), int(parts[1])


async def main(prompt: str, canvas_size: tuple[int, int]) -> None:
    agent = SearchAgent()
    client = PlaceholderAnalysisClient()
    pipeline = Pipeline(search_agent=agent, analysis_client=client)
    await pipeline.run(prompt, canvas_size=canvas_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an artistic collage from Internet Archive material.")
    parser.add_argument("prompt", help="Creative text prompt describing the desired collage")
    parser.add_argument("--canvas", type=parse_canvas, default="1024x1024", help="Canvas size as WIDTHxHEIGHT (default: 1024x1024)")
    args = parser.parse_args()

    asyncio.run(main(args.prompt, args.canvas))
