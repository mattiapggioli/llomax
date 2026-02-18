from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class SearchResult:
    """A single image result returned by the Internet Archive search.

    Attributes:
        identifier: Unique Internet Archive item identifier.
        title: Human-readable title of the item.
        thumbnail_url: URL to a low-resolution preview image.
        details_url: URL to the item's detail page on archive.org.
        image: Downloaded thumbnail as a PIL Image, or ``None`` if not
            yet fetched or if the download failed.
    """

    identifier: str
    title: str
    thumbnail_url: str
    details_url: str
    description: str = ""
    year: str = ""
    image: Image.Image | None = None


@dataclass
class EntityItem:
    """A detected entity crop extracted from an Internet Archive image.

    Attributes:
        item_id: Unique identifier for this crop, formed as
            ``"{parent_image_id}_{label}_{index}"``.
        parent_image_id: Internet Archive identifier of the source image.
        size: ``(width, height)`` in pixels of the cropped region.
        label: Detection class label (e.g. ``"person"``, ``"chair"``).
        metadata: Key fields from the parent image: ``title``, ``year``,
            ``archive_url``, and ``creator``.
        file_path: Local path to the saved crop file, or ``None`` if
            the crop has not been persisted to disk.
        image: In-memory crop as a PIL Image, or ``None`` if not loaded.
            Populated directly by analysis clients to avoid redundant disk
            reads during composition.
    """

    item_id: str
    parent_image_id: str
    size: tuple[int, int]
    label: str
    metadata: dict
    file_path: Path | None = None
    image: Image.Image | None = None

    def load_image(self) -> Image.Image | None:
        """Return the crop as a PIL Image, loading from disk if needed.

        Returns:
            The in-memory image if set, otherwise the image loaded from
            ``file_path``, or ``None`` if neither is available.
        """
        if self.image is not None:
            return self.image
        if self.file_path is not None and self.file_path.exists():
            return Image.open(self.file_path)
        return None


@dataclass
class AnalysisResult:
    """A cropped visual element extracted from a search result image.

    Attributes:
        source_identifier: The ``SearchResult.identifier`` this element
            was extracted from.
        label: NER entity label describing the element (e.g. "person",
            "building"). Placeholder implementations use "unknown".
        image: The cropped region as a PIL Image.
    """

    source_identifier: str
    label: str
    image: Image.Image


@dataclass
class CollageOutput:
    """Final composed collage image with its dimensions and provenance.

    Attributes:
        image: The composed collage as a PIL Image.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        entity_provenance: Metadata records for each entity crop placed
            in the collage, preserving the link to the source IA item.
            Empty list when provenance tracking is unavailable.
    """

    image: Image.Image
    width: int
    height: int
    entity_provenance: list[dict] = field(default_factory=list)
