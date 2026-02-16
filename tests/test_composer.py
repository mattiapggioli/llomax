from PIL import Image

from llomax.composition.composer import compose
from llomax.models import AnalysisResult


def _make_element(width: int = 50, height: int = 50) -> AnalysisResult:
    return AnalysisResult(
        source_identifier="test",
        label="unknown",
        image=Image.new("RGB", (width, height), "blue"),
    )


def test_compose_empty_elements():
    result = compose([], canvas_size=(800, 600))
    assert result.width == 800
    assert result.height == 600
    assert result.image.size == (800, 600)


def test_compose_single_element():
    result = compose([_make_element()], canvas_size=(200, 200))
    assert result.image.size == (200, 200)


def test_compose_multiple_elements():
    elements = [_make_element() for _ in range(5)]
    result = compose(elements, canvas_size=(512, 512))
    assert result.width == 512
    assert result.height == 512


def test_compose_element_larger_than_canvas():
    big = _make_element(width=300, height=300)
    result = compose([big], canvas_size=(100, 100))
    assert result.image.size == (100, 100)
