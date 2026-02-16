from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class SearchResult:
    identifier: str
    title: str
    thumbnail_url: str
    details_url: str
    image: Image.Image | None = None


@dataclass
class AnalysisResult:
    source_identifier: str
    label: str
    image: Image.Image


@dataclass
class CollageOutput:
    image: Image.Image
    width: int
    height: int
