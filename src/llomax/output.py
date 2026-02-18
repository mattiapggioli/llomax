from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from llomax.models import CollageOutput, SearchResult


def save_run(
    collage: CollageOutput,
    search_results: list[SearchResult],
    prompt: str,
    canvas_size: tuple[int, int],
    output_dir: str | Path,
) -> Path:
    """Save a collage and metadata to a timestamped subdirectory.

    Creates ``{output_dir}/{YYYY-MM-DD_HH-MM-SS}/`` containing
    ``collage.png`` and ``metadata.json``. The metadata includes the
    source images used and the provenance of each entity crop placed
    in the collage.

    Args:
        collage: The composed collage output.
        search_results: Source images downloaded during the pipeline run.
        prompt: The original search prompt.
        canvas_size: ``(width, height)`` of the canvas.
        output_dir: Base directory for pipeline outputs.

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now()
    dir_name = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir) / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    collage.image.save(run_dir / "collage.png")

    metadata = {
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt": prompt,
        "canvas_size": list(canvas_size),
        "sources": [
            {
                "identifier": r.identifier,
                "title": r.title,
                "thumbnail_url": r.thumbnail_url,
                "details_url": r.details_url,
            }
            for r in search_results
        ],
        "entity_crops": collage.entity_provenance,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return run_dir
