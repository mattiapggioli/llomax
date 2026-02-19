from __future__ import annotations

import pytest
from pathlib import Path
from PIL import Image

from llomax.analysis.client import PlaceholderAnalysisClient
from llomax.models import SourceImage


@pytest.fixture
def sample_sources(tmp_path: Path) -> list[SourceImage]:
    sources = []
    for i in range(3):
        img = Image.new("RGB", (100, 100), "red")
        path = tmp_path / f"img{i}.jpg"
        img.save(path)
        sources.append(
            SourceImage(
                external_id=f"img{i}",
                title=f"Image {i}",
                description="",
                local_path=path,
                metadata={"year": "1900", "creator": "", "thumbnail_url": "", "details_url": ""},
            )
        )
    return sources


async def test_placeholder_returns_all_images(sample_sources):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_sources)
    assert len(results) == 3


async def test_placeholder_labels_unknown(sample_sources):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_sources)
    assert all(r.label == "unknown" for r in results)


async def test_placeholder_preserves_source_ids(sample_sources):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_sources)
    assert [r.source_id for r in results] == ["img0", "img1", "img2"]


async def test_placeholder_returns_rgba_images(sample_sources):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_sources)
    for fragment in results:
        assert fragment.image_rgba.mode == "RGBA"


async def test_placeholder_bounding_box_covers_full_image(sample_sources):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_sources)
    for fragment in results:
        x1, y1, x2, y2 = fragment.bounding_box
        assert x1 == 0 and y1 == 0
        assert x2 == 100 and y2 == 100


async def test_placeholder_skips_sources_without_local_path():
    client = PlaceholderAnalysisClient()
    source = SourceImage(
        external_id="no_path",
        title="No Path",
        description="",
        local_path=None,
        metadata={},
    )
    results = await client.analyze([source])
    assert results == []


async def test_placeholder_skips_missing_file(tmp_path):
    client = PlaceholderAnalysisClient()
    source = SourceImage(
        external_id="gone",
        title="Gone",
        description="",
        local_path=tmp_path / "nonexistent.jpg",
        metadata={},
    )
    results = await client.analyze([source])
    assert results == []
