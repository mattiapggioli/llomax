from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from llomax.models import CollageOutput, SourceImage


def save_run(
    collage: CollageOutput,
    sources: list[SourceImage],
    prompt: str,
    canvas_size: tuple[int, int],
    output_dir: str | Path,
    run_dir: Path | None = None,
) -> Path:
    """Save a collage and metadata to a timestamped subdirectory.

    Creates ``{output_dir}/{YYYY-MM-DD_HH-MM-SS}/`` containing
    ``collage.png`` and ``metadata.json``. The metadata includes the
    source images used and the provenance of each fragment placed in
    the collage.

    Args:
        collage: The composed collage output.
        sources: Source images downloaded during the pipeline run.
        prompt: The original search prompt.
        canvas_size: ``(width, height)`` of the canvas.
        output_dir: Base directory for pipeline outputs.
        run_dir: Pre-created run directory. When provided, ``output_dir``
            is ignored for directory creation; the caller is responsible
            for creating the directory before calling this function.

    Returns:
        Path to the created run directory.
    """
    if run_dir is None:
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
                "external_id": s.external_id,
                "title": s.title,
                "thumbnail_url": s.metadata.get("thumbnail_url", ""),
                "details_url": s.metadata.get("details_url", ""),
            }
            for s in sources
        ],
        "fragments": collage.fragment_provenance,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return run_dir
