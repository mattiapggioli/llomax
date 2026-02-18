from PIL import Image

from llomax.composition.composer import compose
from llomax.models import EntityItem


def _make_entity(width: int = 50, height: int = 50) -> EntityItem:
    return EntityItem(
        item_id="src_unknown_0",
        parent_image_id="src",
        size=(width, height),
        label="unknown",
        metadata={},
        image=Image.new("RGB", (width, height), "blue"),
    )


def test_compose_empty_elements():
    result = compose([], canvas_size=(800, 600))
    assert result.width == 800
    assert result.height == 600
    assert result.image.size == (800, 600)


def test_compose_single_element():
    result = compose([_make_entity()], canvas_size=(200, 200))
    assert result.image.size == (200, 200)


def test_compose_multiple_elements():
    elements = [_make_entity() for _ in range(5)]
    result = compose(elements, canvas_size=(512, 512))
    assert result.width == 512
    assert result.height == 512


def test_compose_element_larger_than_canvas():
    big = _make_entity(width=300, height=300)
    result = compose([big], canvas_size=(100, 100))
    assert result.image.size == (100, 100)


def test_compose_records_provenance():
    elements = [_make_entity(), _make_entity()]
    result = compose(elements, canvas_size=(200, 200))
    assert len(result.entity_provenance) == 2
    for record in result.entity_provenance:
        assert "item_id" in record
        assert "parent_image_id" in record
        assert "label" in record
        assert "position" in record


def test_compose_with_background():
    bg = Image.new("RGB", (400, 400), "green")
    result = compose([], canvas_size=(200, 200), background=bg)
    assert result.image.size == (200, 200)
    # Background resized; green pixel should not be the default white
    pixel = result.image.getpixel((0, 0))
    assert pixel == (0, 128, 0)


def test_compose_skips_entity_without_image():
    entity_no_image = EntityItem(
        item_id="no_img_unknown_0",
        parent_image_id="no_img",
        size=(50, 50),
        label="unknown",
        metadata={},
        image=None,
        file_path=None,
    )
    result = compose([entity_no_image], canvas_size=(200, 200))
    assert result.entity_provenance == []
