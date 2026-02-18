"""Tests for the output module (save_run)."""

from __future__ import annotations

import json

from PIL import Image

from llomax.models import CollageOutput, SearchResult
from llomax.output import save_run


def _make_collage(width: int = 100, height: int = 80) -> CollageOutput:
    return CollageOutput(
        image=Image.new("RGB", (width, height), "white"), width=width, height=height
    )


def _make_results() -> list[SearchResult]:
    return [
        SearchResult(
            identifier="item-1",
            title="First Item",
            thumbnail_url="https://archive.org/services/img/item-1",
            details_url="https://archive.org/details/item-1",
        ),
        SearchResult(
            identifier="item-2",
            title="Second Item",
            thumbnail_url="https://archive.org/services/img/item-2",
            details_url="https://archive.org/details/item-2",
        ),
    ]


def test_save_run_creates_collage_and_metadata(tmp_path):
    collage = _make_collage()
    results = _make_results()
    prompt = "vintage botanical illustrations"
    canvas_size = (100, 80)

    run_dir = save_run(collage, results, prompt, canvas_size, tmp_path)

    assert run_dir.parent == tmp_path
    assert (run_dir / "collage.png").is_file()
    assert (run_dir / "metadata.json").is_file()


def test_save_run_metadata_content(tmp_path):
    collage = _make_collage()
    results = _make_results()
    prompt = "test prompt"
    canvas_size = (1920, 1080)

    run_dir = save_run(collage, results, prompt, canvas_size, tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert metadata["prompt"] == "test prompt"
    assert metadata["canvas_size"] == [1920, 1080]
    assert "timestamp" in metadata
    assert len(metadata["sources"]) == 2
    assert metadata["sources"][0]["identifier"] == "item-1"
    assert metadata["sources"][1]["title"] == "Second Item"


def test_save_run_collage_image_dimensions(tmp_path):
    collage = _make_collage(200, 150)
    run_dir = save_run(collage, [], "prompt", (200, 150), tmp_path)

    saved = Image.open(run_dir / "collage.png")
    assert saved.size == (200, 150)


def test_save_run_directory_name_format(tmp_path):
    collage = _make_collage()
    run_dir = save_run(collage, [], "prompt", (100, 100), tmp_path)

    # Directory name should match YYYY-MM-DD_HH-MM-SS format
    name = run_dir.name
    parts = name.split("_")
    assert len(parts) == 2
    date_parts = parts[0].split("-")
    time_parts = parts[1].split("-")
    assert len(date_parts) == 3  # YYYY, MM, DD
    assert len(time_parts) == 3  # HH, MM, SS


def test_save_run_empty_results(tmp_path):
    collage = _make_collage()
    run_dir = save_run(collage, [], "empty search", (100, 100), tmp_path)

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["sources"] == []


def test_save_run_returns_path(tmp_path):
    collage = _make_collage()
    run_dir = save_run(collage, [], "prompt", (100, 100), tmp_path)

    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path


def test_save_run_includes_entity_crops(tmp_path):
    provenance = [
        {
            "item_id": "img_person_0",
            "parent_image_id": "img",
            "label": "person",
            "size": [80, 120],
            "position": [10, 20],
            "metadata": {"title": "Test", "year": "1900", "archive_url": "", "creator": ""},
        }
    ]
    collage = CollageOutput(
        image=Image.new("RGB", (100, 100), "white"),
        width=100,
        height=100,
        entity_provenance=provenance,
    )
    run_dir = save_run(collage, [], "prompt", (100, 100), tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert "entity_crops" in metadata
    assert len(metadata["entity_crops"]) == 1
    assert metadata["entity_crops"][0]["item_id"] == "img_person_0"
    assert metadata["entity_crops"][0]["label"] == "person"
