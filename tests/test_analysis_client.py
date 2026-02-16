import pytest
from PIL import Image

from llomax.analysis.client import PlaceholderAnalysisClient
from llomax.models import SearchResult


@pytest.fixture
def sample_results():
    return [
        SearchResult(
            identifier=f"img{i}",
            title=f"Image {i}",
            thumbnail_url=f"https://archive.org/services/img/img{i}",
            details_url=f"https://archive.org/details/img{i}",
            image=Image.new("RGB", (100, 100), "red"),
        )
        for i in range(3)
    ]


async def test_placeholder_returns_all_images(sample_results):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_results)
    assert len(results) == 3


async def test_placeholder_labels_unknown(sample_results):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_results)
    assert all(r.label == "unknown" for r in results)


async def test_placeholder_preserves_identifiers(sample_results):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_results)
    assert [r.source_identifier for r in results] == ["img0", "img1", "img2"]


async def test_placeholder_preserves_images(sample_results):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_results)
    for orig, analyzed in zip(sample_results, results):
        assert analyzed.image is orig.image


async def test_placeholder_skips_results_without_image():
    client = PlaceholderAnalysisClient()
    results = await client.analyze(
        [
            SearchResult(
                identifier="no_img",
                title="No Image",
                thumbnail_url="",
                details_url="",
                image=None,
            ),
        ]
    )
    assert results == []
