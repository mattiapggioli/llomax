from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

import numpy as np
from loguru import logger
from PIL import Image

from llomax.models import Fragment, SourceImage

if TYPE_CHECKING:
    pass


class AnalysisClient(Protocol):
    """Protocol for image segmentation backends.

    Implementations receive source images and return extracted visual
    segments as ``Fragment`` objects with transparent RGBA backgrounds.
    """

    async def analyze(self, sources: list[SourceImage]) -> list[Fragment]:
        """Segment source images and return extracted fragments.

        Args:
            sources: Source images with valid ``local_path`` values.

        Returns:
            A list of ``Fragment`` objects extracted from the images.
        """
        ...


class PlaceholderAnalysisClient:
    """Passthrough client that returns each source image as a single full-frame fragment.

    Useful for testing the pipeline end-to-end before a SAM model checkpoint
    is available. No model is loaded; each downloaded source image is wrapped
    as a single full-image ``Fragment`` with label ``"unknown"``.
    """

    async def analyze(self, sources: list[SourceImage]) -> list[Fragment]:
        """Return each source image as a full-frame fragment with label ``"unknown"``.

        Args:
            sources: Source images to pass through. Sources without a
                valid ``local_path`` are silently skipped.

        Returns:
            One ``Fragment`` per source image that loads successfully.
        """
        return [
            f
            for source in sources
            if (f := self._fragment_from_source(source)) is not None
        ]

    def _fragment_from_source(self, source: SourceImage) -> Fragment | None:
        """Create a full-frame ``Fragment`` from a source image.

        Args:
            source: Source image to wrap as a fragment.

        Returns:
            Full-frame RGBA ``Fragment``, or ``None`` if the image cannot be loaded.
        """
        img = source.load_image()
        if img is None:
            return None
        rgba = img.convert("RGBA")
        w, h = rgba.size
        return Fragment(
            source_id=source.external_id,
            image_rgba=rgba,
            bounding_box=(0, 0, w, h),
        )


class YoloAnalysisClient:
    """YOLO-based instance segmentation client.

    Uses an Ultralytics segmentation model (e.g. ``yolov8n-seg.pt``) to detect
    and extract visual instances from source images. Each detected instance
    becomes a ``Fragment`` with a transparent RGBA background shaped by the
    segmentation mask — no rectangular black borders.

    Args:
        model_name: Ultralytics model name or path. Requires a segmentation
            model (suffix ``-seg.pt``).
        min_conf: Minimum detection confidence threshold (0–1).
    """

    def __init__(self, model_name: str = "yolov8n-seg.pt", min_conf: float = 0.25) -> None:
        self._model_name = model_name
        self._min_conf = min_conf
        self._model = None

    def _get_model(self):
        """Return the cached YOLO model, loading it on first access."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name)
        return self._model

    async def analyze(self, sources: list[SourceImage]) -> list[Fragment]:
        """Run instance segmentation on all source images.

        Each source is processed in a thread to avoid blocking the event loop.

        Args:
            sources: Source images with valid ``local_path`` values.

        Returns:
            All fragments extracted across all source images.
        """
        fragments: list[Fragment] = []
        for source in sources:
            source_fragments = await asyncio.to_thread(self._segment_source, source)
            fragments.extend(source_fragments)
        return fragments

    def _segment_source(self, source: SourceImage) -> list[Fragment]:
        """Extract instance segments from a single source image.

        Args:
            source: Source image with a valid ``local_path``.

        Returns:
            List of RGBA ``Fragment`` objects, one per detected instance.
            Empty list if the image cannot be loaded or no instances pass
            the confidence threshold.
        """
        img = source.load_image()
        if img is None:
            logger.warning("Cannot segment {}: local_path unavailable.", source.external_id)
            return []

        rgb_img = img.convert("RGB")
        model = self._get_model()
        results = model(rgb_img, conf=self._min_conf)

        fragments: list[Fragment] = []
        for result in results:
            if result.masks is None:
                continue

            rgb_array = np.array(rgb_img)
            orig_h, orig_w = result.orig_shape

            for mask_tensor, xyxy, cls_tensor in zip(
                result.masks.data,
                result.boxes.xyxy,
                result.boxes.cls,
            ):
                fragment = self._build_fragment(
                    mask_tensor,
                    xyxy,
                    cls_tensor,
                    rgb_array,
                    orig_h,
                    orig_w,
                    result.names,
                    source.external_id,
                )
                if fragment is not None:
                    fragments.append(fragment)

        return fragments

    def _build_fragment(
        self,
        mask_tensor,
        xyxy,
        cls_tensor,
        rgb_array: np.ndarray,
        orig_h: int,
        orig_w: int,
        class_names: dict,
        source_id: str,
    ) -> Fragment | None:
        """Convert a YOLO mask into a transparent-background RGBA ``Fragment``.

        Args:
            mask_tensor: Float32 tensor of shape ``(H_inf, W_inf)`` with values in [0, 1].
            xyxy: Tensor with ``(x1, y1, x2, y2)`` in original image pixel coordinates.
            cls_tensor: Scalar class-index tensor.
            rgb_array: Full source image as a ``(H, W, 3)`` uint8 NumPy array.
            orig_h: Original image height in pixels.
            orig_w: Original image width in pixels.
            class_names: Dict mapping integer class index to class name string.
            source_id: ``SourceImage.external_id`` of the parent image.

        Returns:
            RGBA ``Fragment`` with transparent background, or ``None`` if the
            bounding box is degenerate (zero area).
        """
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.shape != (orig_h, orig_w):
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize((orig_w, orig_h), Image.BILINEAR)
            binary_mask = np.array(mask_pil) > 127
        else:
            binary_mask = mask_np > 0.5

        x1, y1, x2, y2 = (int(v) for v in xyxy.cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop_rgb = rgb_array[y1:y2, x1:x2]
        crop_mask = binary_mask[y1:y2, x1:x2]

        rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
        rgba[..., :3] = crop_rgb
        rgba[..., 3] = np.where(crop_mask, 255, 0).astype(np.uint8)

        cls_idx = int(cls_tensor.item())
        label = class_names.get(cls_idx, f"class_{cls_idx}")

        return Fragment(
            source_id=source_id,
            image_rgba=Image.fromarray(rgba, mode="RGBA"),
            bounding_box=(x1, y1, x2, y2),
            label=label,
        )
