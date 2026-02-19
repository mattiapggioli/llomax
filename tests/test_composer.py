from __future__ import annotations

from PIL import Image

from llomax.composition.composer import compose
from llomax.models import Fragment


def _make_fragment(width: int = 50, height: int = 50) -> Fragment:
    rgba = Image.new("RGBA", (width, height), (0, 0, 255, 255))
    return Fragment(
        source_id="src",
        image_rgba=rgba,
        bounding_box=(0, 0, width, height),
        label="unknown",
    )


def test_compose_empty_elements():
    result = compose([], canvas_size=(800, 600))
    assert result.width == 800
    assert result.height == 600
    assert result.image.size == (800, 600)


def test_compose_single_element():
    result = compose([_make_fragment()], canvas_size=(200, 200))
    assert result.image.size == (200, 200)


def test_compose_multiple_elements():
    fragments = [_make_fragment() for _ in range(5)]
    result = compose(fragments, canvas_size=(512, 512))
    assert result.width == 512
    assert result.height == 512


def test_compose_element_larger_than_canvas():
    big = _make_fragment(width=300, height=300)
    result = compose([big], canvas_size=(100, 100))
    assert result.image.size == (100, 100)


def test_compose_records_provenance():
    fragments = [_make_fragment(), _make_fragment()]
    result = compose(fragments, canvas_size=(200, 200))
    assert len(result.fragment_provenance) == 2
    for record in result.fragment_provenance:
        assert "source_id" in record
        assert "bounding_box" in record
        assert "label" in record
        assert "position" in record


def test_compose_with_background():
    bg = Image.new("RGB", (400, 400), "green")
    result = compose([], canvas_size=(200, 200), background=bg)
    assert result.image.size == (200, 200)
    # Background resized and preserved; green pixel should not be default white
    pixel = result.image.getpixel((0, 0))
    assert pixel == (0, 128, 0)


def test_compose_output_is_rgb():
    result = compose([_make_fragment()], canvas_size=(200, 200))
    assert result.image.mode == "RGB"


def test_compose_alpha_transparency_respected():
    # Fully transparent fragment should not cover the white canvas
    rgba_transparent = Image.new("RGBA", (50, 50), (255, 0, 0, 0))
    fragment = Fragment(
        source_id="src",
        image_rgba=rgba_transparent,
        bounding_box=(0, 0, 50, 50),
    )
    result = compose([fragment], canvas_size=(200, 200))
    # Canvas stays white because fragment is fully transparent
    pixel = result.image.getpixel((0, 0))
    assert pixel == (255, 255, 255)
