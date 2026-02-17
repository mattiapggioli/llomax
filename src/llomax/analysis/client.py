from __future__ import annotations

from typing import Protocol

from llomax.models import AnalysisResult, SearchResult


class AnalysisClient(Protocol):
    """Protocol for image analysis backends.

    Implementations receive search results and return extracted visual
    elements with NER labels.
    """

    async def analyze(self, images: list[SearchResult]) -> list[AnalysisResult]:
        """Analyze images and extract labelled visual elements.

        Args:
            images: Search results with downloaded thumbnail images.

        Returns:
            A list of cropped elements with entity labels.
        """
        ...


class PlaceholderAnalysisClient:
    """Passthrough client that returns each image as-is with label "unknown".

    Useful for testing the pipeline end-to-end before a real NER backend
    is available.
    """

    async def analyze(self, images: list[SearchResult]) -> list[AnalysisResult]:
        """Return each image unmodified with the label "unknown".

        Args:
            images: Search results to pass through. Results without an
                image are silently skipped.

        Returns:
            One ``AnalysisResult`` per input image that has a non-None image.
        """
        return [
            AnalysisResult(
                source_identifier=img.identifier,
                label="unknown",
                image=img.image,
            )
            for img in images
            if img.image is not None
        ]
