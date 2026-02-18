from __future__ import annotations

import os
from typing import Callable

import anthropic

from llomax.analysis.client import AnalysisClient
from llomax.composition.composer import compose as default_compose
from llomax.models import AnalysisResult, CollageOutput, SearchResult
from llomax.output import save_run
from llomax.search.clients.internet_archive_client import ImageResult
from llomax.search.curator import select_assets
from llomax.search.internet_archive_agent import InternetArchiveAgent
from llomax.search.thumbnails import download_thumbnails


class Pipeline:
    """Multi-stage pipeline: discovery -> curation -> analysis -> composition."""

    def __init__(
        self,
        search_agent: InternetArchiveAgent,
        analysis_client: AnalysisClient,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        compose_fn: Callable[
            [list[AnalysisResult], tuple[int, int]], CollageOutput
        ] = default_compose,
    ) -> None:
        """Initialize the pipeline.

        Args:
            search_agent: Agent for discovering images on the Internet Archive.
            analysis_client: Backend for extracting visual elements from images.
            anthropic_client: Anthropic async client for the curator stage.
                Defaults to the search agent's client.
            compose_fn: Callable that arranges elements onto a canvas.
        """
        self.search_agent = search_agent
        self.analysis_client = analysis_client
        self.anthropic_client = anthropic_client or search_agent.client
        self.compose_fn = compose_fn

    async def run(
        self,
        prompt: str,
        canvas_size: tuple[int, int] = (1024, 1024),
        max_items: int = 20,
    ) -> CollageOutput:
        """Execute the full pipeline from prompt to collage."""
        # Stage 1a: Plan searches — the LLM only registers intents, sees no raw results.
        search_plan = await self.search_agent.plan_search(prompt, max_items=max_items)

        # Stage 1b: Execute plan directly in Python, keeping raw data out of LLM context.
        raw_results = self._execute_search_plan(search_plan)

        # Stage 2: Curation with sanitized candidates.
        candidates = self._sanitize(raw_results)
        selected_ids = await select_assets(
            prompt, candidates, self.anthropic_client, max_items=max_items
        )
        search_results = self._build_search_results(raw_results, selected_ids)

        await download_thumbnails(search_results)

        elements = await self.analysis_client.analyze(search_results)
        collage = self.compose_fn(elements, canvas_size)

        output_dir = os.environ.get("OUTPUT_DIR", "output")
        save_run(collage, search_results, prompt, canvas_size, output_dir)

        return collage

    def _execute_search_plan(self, plan: list[dict]) -> list[ImageResult]:
        """Execute each item in the search plan and return deduplicated results.

        Args:
            plan: List of search plan items from plan_search.

        Returns:
            Deduplicated list of ImageResult items, keyed by identifier.
        """
        seen: dict[str, ImageResult] = {}
        for item in plan:
            kwargs: dict = {
                "keywords": item["keywords"],
                "collection": item.get("collection"),
                "date_filter": item.get("date_filter"),
            }
            if item.get("max_results") is not None:
                kwargs["max_results"] = item["max_results"]
            for result in self.search_agent.ia_client.search_images(**kwargs):
                ident = result.get("identifier", "")
                if ident and ident not in seen:
                    seen[ident] = result
        return list(seen.values())

    def _sanitize(self, raw_results: list[ImageResult]) -> list[dict]:
        """Deduplicate raw results and extract curator-relevant fields.

        Args:
            raw_results: Raw image results from the search agent.

        Returns:
            Deduplicated list of dicts with identifier, title, description, year.
        """
        seen: dict[str, dict] = {}
        for item in raw_results:
            ident = item.get("identifier", "")
            if not ident or ident in seen:
                continue
            seen[ident] = {
                "identifier": ident,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "year": item.get("date", "")[:4] if item.get("date") else "",
            }
        return list(seen.values())

    def _build_search_results(
        self, raw_results: list[ImageResult], selected_ids: list[str]
    ) -> list[SearchResult]:
        """Build deduplicated SearchResult objects for the selected identifiers.

        Args:
            raw_results: Raw image results from the search agent.
            selected_ids: Identifiers chosen by the curator.

        Returns:
            Deduplicated list of SearchResult objects.
        """
        selected_set = set(selected_ids)
        seen_ids: set[str] = set()
        results: list[SearchResult] = []
        for item in raw_results:
            ident = item.get("identifier", "")
            if ident not in selected_set or ident in seen_ids:
                continue
            seen_ids.add(ident)
            results.append(
                SearchResult(
                    identifier=ident,
                    title=item.get("title", ""),
                    thumbnail_url=f"https://archive.org/services/img/{ident}",
                    details_url=f"https://archive.org/details/{ident}",
                    description=item.get("description", ""),
                    year=item.get("date", "")[:4] if item.get("date") else "",
                )
            )
        return results
