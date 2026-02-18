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
            year="1900",
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
    assert [r.parent_image_id for r in results] == ["img0", "img1", "img2"]


async def test_placeholder_item_ids_derived_from_identifier(sample_results):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_results)
    for r in results:
        assert r.item_id.startswith(r.parent_image_id)


async def test_placeholder_preserves_images(sample_results):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_results)
    for orig, entity in zip(sample_results, results):
        assert entity.image is orig.image


async def test_placeholder_metadata_includes_title_and_year(sample_results):
    client = PlaceholderAnalysisClient()
    results = await client.analyze(sample_results)
    for i, entity in enumerate(results):
        assert entity.metadata["title"] == f"Image {i}"
        assert entity.metadata["year"] == "1900"


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
