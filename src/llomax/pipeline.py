from __future__ import annotations

from typing import Callable

from llomax.analysis.client import AnalysisClient
from llomax.composition.composer import compose as default_compose
from llomax.models import AnalysisResult, CollageOutput
from llomax.search.agent import SearchAgent


class Pipeline:
    def __init__(
        self,
        search_agent: SearchAgent,
        analysis_client: AnalysisClient,
        compose_fn: Callable[[list[AnalysisResult], tuple[int, int]], CollageOutput] = default_compose,
    ) -> None:
        self.search_agent = search_agent
        self.analysis_client = analysis_client
        self.compose_fn = compose_fn

    async def run(
        self,
        prompt: str,
        canvas_size: tuple[int, int] = (1024, 1024),
    ) -> CollageOutput:
        results = await self.search_agent.search(prompt)
        elements = await self.analysis_client.analyze(results)
        return self.compose_fn(elements, canvas_size)
