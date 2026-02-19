from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class SourceImage:
    """An Internet Archive source image with local file reference.

    Attributes:
        external_id: Unique Internet Archive item identifier.
        title: Human-readable title of the item.
        description: Text description of the item from the archive.
        local_path: Local filesystem path to the downloaded image file,
            or ``None`` if the image has not been downloaded.
        metadata: Complete metadata dictionary from the Internet Archive
            record, including fields such as ``creator``, ``year``,
            ``thumbnail_url``, ``details_url``, and ``collection``.
    """

    external_id: str
    title: str
    description: str
    local_path: Path | None
    metadata: dict

    def load_image(self) -> Image.Image | None:
        """Return the image loaded from ``local_path``.

        Returns:
            PIL Image loaded from disk, or ``None`` if ``local_path``
            is unset or the file does not exist.
        """
        if self.local_path is None or not self.local_path.exists():
            return None
        return Image.open(self.local_path)


@dataclass
class Fragment:
    """A visual segment extracted from a source image.

    Attributes:
        source_id: ``SourceImage.external_id`` of the parent image.
        image_rgba: The segmented region as an RGBA PIL Image with a
            transparent background preserving the segment shape.
        bounding_box: ``(x1, y1, x2, y2)`` pixel coordinates of the
            segment within the source image.
        label: Short entity label for the segment (e.g. ``"person"``,
            ``"object"``). Populated by the annotation stage.
        description: Free-text description of the segment content.
            Populated by the annotation stage.
    """

    source_id: str
    image_rgba: Image.Image
    bounding_box: tuple[int, int, int, int]
    label: str = "unknown"
    description: str = ""


@dataclass
class CollageOutput:
    """Final composed collage image with its dimensions and provenance.

    Attributes:
        image: The composed collage as a PIL Image.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        fragment_provenance: Metadata records for each fragment placed
            in the collage, preserving the link to the source IA item.
            Empty list when provenance tracking is unavailable.
    """

    image: Image.Image
    width: int
    height: int
    fragment_provenance: list[dict] = field(default_factory=list)
