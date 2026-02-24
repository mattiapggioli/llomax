from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from llomax.analysis.client import YoloAnalysisClient
from llomax.models import SourceImage


def _make_source(
    tmp_path: Path, name: str = "img", size: tuple[int, int] = (100, 100)
) -> SourceImage:
    img = Image.new("RGB", size, color=(128, 64, 32))
    path = tmp_path / f"{name}.jpg"
    img.save(path)
    return SourceImage(
        external_id=name,
        title=name,
        description="",
        local_path=path,
        metadata={},
    )


def _make_yolo_result(
    orig_h: int = 100,
    orig_w: int = 100,
    num_detections: int = 1,
    class_names: dict | None = None,
) -> MagicMock:
    """Build a mock YOLO result with masks, boxes, and class information."""
    if class_names is None:
        class_names = {0: "person"}

    result = MagicMock()
    result.orig_shape = (orig_h, orig_w)
    result.names = class_names

    if num_detections == 0:
        result.masks = None
        return result

    # masks.data: (N, H_inf, W_inf) — use same size as original for simplicity
    mask_data = []
    for _ in range(num_detections):
        m = torch.zeros(orig_h, orig_w)
        m[10:50, 10:50] = 1.0  # filled square mask region
        mask_data.append(m)
    result.masks = MagicMock()
    result.masks.data = torch.stack(mask_data)

    # boxes.xyxy: (N, 4) in original image pixels
    xyxy_data = torch.tensor([[10.0, 10.0, 50.0, 50.0]] * num_detections)
    result.boxes = MagicMock()
    result.boxes.xyxy = xyxy_data

    # boxes.cls: (N,) class indices
    result.boxes.cls = torch.zeros(num_detections)

    return result


@pytest.fixture
def sample_source(tmp_path):
    return _make_source(tmp_path)


def _patched_client(result: MagicMock) -> YoloAnalysisClient:
    """Return a YoloAnalysisClient whose model always returns ``result``."""
    client = YoloAnalysisClient(model_name="yolov8n-seg.pt")
    mock_model = MagicMock()
    mock_model.return_value = [result]
    mock_model.names = result.names if result.masks is not None else {}
    client._model = mock_model
    return client


async def test_yolo_returns_fragments_for_detected_instances(sample_source):
    result = _make_yolo_result(num_detections=2)
    client = _patched_client(result)
    fragments = await client.analyze([sample_source])
    assert len(fragments) == 2


async def test_yolo_fragment_mode_is_rgba(sample_source):
    result = _make_yolo_result(num_detections=1)
    client = _patched_client(result)
    fragments = await client.analyze([sample_source])
    for fragment in fragments:
        assert fragment.image_rgba.mode == "RGBA"


async def test_yolo_fragment_label_matches_class_name(sample_source):
    result = _make_yolo_result(num_detections=1, class_names={0: "cat"})
    client = _patched_client(result)
    fragments = await client.analyze([sample_source])
    assert fragments[0].label == "cat"


async def test_yolo_fragment_source_id_matches(sample_source):
    result = _make_yolo_result(num_detections=1)
    client = _patched_client(result)
    fragments = await client.analyze([sample_source])
    assert fragments[0].source_id == sample_source.external_id


async def test_yolo_fragment_bounding_box(sample_source):
    result = _make_yolo_result(num_detections=1)
    client = _patched_client(result)
    fragments = await client.analyze([sample_source])
    x1, y1, x2, y2 = fragments[0].bounding_box
    assert x1 == 10 and y1 == 10 and x2 == 50 and y2 == 50


async def test_yolo_mask_applied_as_alpha(sample_source):
    """Pixels outside the mask must have alpha=0; inside must have alpha=255."""
    result = _make_yolo_result(orig_h=100, orig_w=100, num_detections=1)
    client = _patched_client(result)
    fragments = await client.analyze([sample_source])
    rgba = np.array(fragments[0].image_rgba)

    # All pixels inside the mask region (relative to crop starting at 10,10)
    # The crop is [10:50, 10:50], mask fills the whole crop → all alpha=255
    assert np.all(rgba[..., 3] == 255)


async def test_yolo_no_masks_returns_empty(tmp_path):
    source = _make_source(tmp_path, "no_mask")
    result = _make_yolo_result(num_detections=0)
    client = _patched_client(result)
    fragments = await client.analyze([source])
    assert fragments == []


async def test_yolo_skips_source_without_local_path():
    client = YoloAnalysisClient()
    source = SourceImage(
        external_id="no_path",
        title="",
        description="",
        local_path=None,
        metadata={},
    )
    fragments = await client.analyze([source])
    assert fragments == []


async def test_yolo_skips_missing_file(tmp_path):
    client = YoloAnalysisClient()
    source = SourceImage(
        external_id="gone",
        title="",
        description="",
        local_path=tmp_path / "nonexistent.jpg",
        metadata={},
    )
    fragments = await client.analyze([source])
    assert fragments == []


async def test_yolo_degenerate_box_is_skipped(tmp_path):
    """A detection whose clamped box has zero area must not produce a fragment."""
    source = _make_source(tmp_path, "degenerate")
    result = MagicMock()
    result.orig_shape = (100, 100)
    result.names = {0: "thing"}

    mask = torch.ones(100, 100)
    result.masks = MagicMock()
    result.masks.data = torch.stack([mask])

    # Box entirely outside the image → after clamping x2 <= x1
    result.boxes = MagicMock()
    result.boxes.xyxy = torch.tensor([[-10.0, -10.0, -5.0, -5.0]])
    result.boxes.cls = torch.zeros(1)

    client = _patched_client(result)
    fragments = await client.analyze([source])
    assert fragments == []


async def test_yolo_aggregates_multiple_sources(tmp_path):
    sources = [_make_source(tmp_path, f"src{i}") for i in range(3)]
    result = _make_yolo_result(num_detections=2)

    client = YoloAnalysisClient(model_name="yolov8n-seg.pt")
    mock_model = MagicMock()
    mock_model.return_value = [result]
    mock_model.names = result.names
    client._model = mock_model

    fragments = await client.analyze(sources)
    assert len(fragments) == 6  # 2 detections × 3 sources
