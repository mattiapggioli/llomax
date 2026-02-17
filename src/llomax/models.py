"""Domain models shared across the llomax pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Final composed collage image with its dimensions.

    Attributes:
        image: The composed collage as a PIL Image.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
    """

    image: Image.Image
    width: int
    height: int
