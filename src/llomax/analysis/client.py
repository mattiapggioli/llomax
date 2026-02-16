from __future__ import annotations

from typing import Protocol

from llomax.models import AnalysisResult, SearchResult


class AnalysisClient(Protocol):
    async def analyze(self, images: list[SearchResult]) -> list[AnalysisResult]: ...


class PlaceholderAnalysisClient:
    """Returns each input image as-is with label 'unknown'."""

    async def analyze(self, images: list[SearchResult]) -> list[AnalysisResult]:
        return [
            AnalysisResult(
                source_identifier=img.identifier,
                label="unknown",
                image=img.image,
            )
            for img in images
            if img.image is not None
        ]
