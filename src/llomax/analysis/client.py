from __future__ import annotations

from typing import Protocol

from llomax.models import Fragment, SourceImage


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
        fragments: list[Fragment] = []
        for source in sources:
            img = source.load_image()
            if img is None:
                continue
            rgba = img.convert("RGBA")
            w, h = rgba.size
            fragments.append(
                Fragment(
                    source_id=source.external_id,
                    image_rgba=rgba,
                    bounding_box=(0, 0, w, h),
                )
            )
        return fragments
