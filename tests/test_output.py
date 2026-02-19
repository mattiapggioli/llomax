"""Tests for the output module (save_run)."""

from __future__ import annotations

import json

from PIL import Image

from llomax.models import CollageOutput, SourceImage
from llomax.output import save_run


def _make_collage(width: int = 100, height: int = 80) -> CollageOutput:
    return CollageOutput(
        image=Image.new("RGB", (width, height), "white"), width=width, height=height
    )


def _make_sources() -> list[SourceImage]:
    return [
        SourceImage(
            external_id="item-1",
            title="First Item",
            description="",
            local_path=None,
            metadata={
                "thumbnail_url": "https://archive.org/services/img/item-1",
                "details_url": "https://archive.org/details/item-1",
            },
        ),
        SourceImage(
            external_id="item-2",
            title="Second Item",
            description="",
            local_path=None,
            metadata={
                "thumbnail_url": "https://archive.org/services/img/item-2",
                "details_url": "https://archive.org/details/item-2",
            },
        ),
    ]


def test_save_run_creates_collage_and_metadata(tmp_path):
    collage = _make_collage()
    sources = _make_sources()
    prompt = "vintage botanical illustrations"
    canvas_size = (100, 80)

    run_dir = save_run(collage, sources, prompt, canvas_size, tmp_path)

    assert run_dir.parent == tmp_path
    assert (run_dir / "collage.png").is_file()
    assert (run_dir / "metadata.json").is_file()


def test_save_run_metadata_content(tmp_path):
    collage = _make_collage()
    sources = _make_sources()
    prompt = "test prompt"
    canvas_size = (1920, 1080)

    run_dir = save_run(collage, sources, prompt, canvas_size, tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert metadata["prompt"] == "test prompt"
    assert metadata["canvas_size"] == [1920, 1080]
    assert "timestamp" in metadata
    assert len(metadata["sources"]) == 2
    assert metadata["sources"][0]["external_id"] == "item-1"
    assert metadata["sources"][1]["title"] == "Second Item"


def test_save_run_collage_image_dimensions(tmp_path):
    collage = _make_collage(200, 150)
    run_dir = save_run(collage, [], "prompt", (200, 150), tmp_path)

    saved = Image.open(run_dir / "collage.png")
    assert saved.size == (200, 150)


def test_save_run_directory_name_format(tmp_path):
    collage = _make_collage()
    run_dir = save_run(collage, [], "prompt", (100, 100), tmp_path)

    name = run_dir.name
    parts = name.split("_")
    assert len(parts) == 2
    date_parts = parts[0].split("-")
    time_parts = parts[1].split("-")
    assert len(date_parts) == 3  # YYYY, MM, DD
    assert len(time_parts) == 3  # HH, MM, SS


def test_save_run_empty_sources(tmp_path):
    collage = _make_collage()
    run_dir = save_run(collage, [], "empty search", (100, 100), tmp_path)

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["sources"] == []


def test_save_run_returns_path(tmp_path):
    collage = _make_collage()
    run_dir = save_run(collage, [], "prompt", (100, 100), tmp_path)

    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path


def test_save_run_includes_fragment_provenance(tmp_path):
    provenance = [
        {
            "source_id": "img",
            "bounding_box": [10, 20, 90, 140],
            "label": "unknown",
            "description": "placeholder",
            "position": [10, 20],
        }
    ]
    collage = CollageOutput(
        image=Image.new("RGB", (100, 100), "white"),
        width=100,
        height=100,
        fragment_provenance=provenance,
    )
    run_dir = save_run(collage, [], "prompt", (100, 100), tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert "fragments" in metadata
    assert len(metadata["fragments"]) == 1
    assert metadata["fragments"][0]["source_id"] == "img"
    assert metadata["fragments"][0]["label"] == "unknown"
