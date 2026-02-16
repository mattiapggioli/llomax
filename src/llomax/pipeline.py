"""End-to-end collage pipeline orchestrating search, analysis, and composition."""

from __future__ import annotations

from typing import Callable

from llomax.analysis.client import AnalysisClient
from llomax.composition.composer import compose as default_compose
from llomax.models import AnalysisResult, CollageOutput
from llomax.search.agent import SearchAgent


class Pipeline:
    """Three-stage pipeline: search -> analysis -> composition.

    Attributes:
        search_agent: Agent that queries the Internet Archive for images.
        analysis_client: Client that performs NER / cropping on images.
        compose_fn: Callable that assembles analysis results into a collage.
    """

    def __init__(
        self,
        search_agent: SearchAgent,
        analysis_client: AnalysisClient,
        compose_fn: Callable[
            [list[AnalysisResult], tuple[int, int]], CollageOutput
        ] = default_compose,
    ) -> None:
        self.search_agent = search_agent
        self.analysis_client = analysis_client
        self.compose_fn = compose_fn

    async def run(
        self,
        prompt: str,
        canvas_size: tuple[int, int] = (1024, 1024),
    ) -> CollageOutput:
        """Execute the full pipeline from prompt to collage.

        Args:
            prompt: Creative text prompt describing the desired collage.
            canvas_size: ``(width, height)`` in pixels for the output canvas.

        Returns:
            A ``CollageOutput`` containing the composed image.
        """
        results = await self.search_agent.search(prompt)
        elements = await self.analysis_client.analyze(results)
        return self.compose_fn(elements, canvas_size)
