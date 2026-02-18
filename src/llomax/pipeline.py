from __future__ import annotations

import os
from typing import Callable

import anthropic
from PIL import Image

from llomax.analysis.client import AnalysisClient
from llomax.composition.composer import compose as default_compose
from llomax.models import CollageOutput, EntityItem, SearchResult
from llomax.output import save_run
from llomax.search.clients.internet_archive_client import ImageResult
from llomax.search.curator import select_assets
from llomax.search.internet_archive_agent import InternetArchiveAgent
from llomax.search.thumbnails import download_thumbnails


class Pipeline:
    """Multi-stage pipeline: discovery -> analysis -> curation -> composition."""

    def __init__(
        self,
        search_agent: InternetArchiveAgent,
        analysis_client: AnalysisClient,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        compose_fn: Callable[
            [list[EntityItem], tuple[int, int], Image.Image | None], CollageOutput
        ] = default_compose,
    ) -> None:
        """Initialize the pipeline.

        Args:
            search_agent: Agent for discovering images on the Internet Archive.
            analysis_client: Backend for extracting visual entity crops from images.
            anthropic_client: Anthropic async client for the curator stage.
                Defaults to the search agent's client.
            compose_fn: Callable that arranges entity crops onto a canvas.
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
        """Execute the full pipeline from prompt to collage.

        Pipeline stages:
        1. ``plan_search`` — LLM registers search intents (no raw data in context).
        2. ``_execute_search_plan`` — Python layer runs each query directly.
        3. Download thumbnails for all candidate images.
        4. ``analysis_client.analyze`` — Detect entities and produce crops.
        5. ``select_assets`` — Curator selects the best entity crops.
        6. ``compose_fn`` — Paste selected crops onto the background.
        7. ``save_run`` — Persist collage and provenance metadata.

        Args:
            prompt: Creative text prompt describing the desired collage.
            canvas_size: ``(width, height)`` in pixels for the output canvas.
            max_items: Target number of entity crops in the final collage.

        Returns:
            The composed ``CollageOutput``.
        """
        # Stage 1: Plan searches without exposing raw results to the LLM.
        search_plan = await self.search_agent.plan_search(prompt, max_items=max_items)

        # Stage 2: Execute plan directly in Python.
        raw_results = self._execute_search_plan(search_plan)

        # Stage 3: Build search result objects and download thumbnails.
        search_results = self._build_search_results_from_raw(raw_results)
        await download_thumbnails(search_results)

        # Stage 4: Detect entities from all downloaded images.
        entity_items = await self.analysis_client.analyze(search_results)

        # Stage 5: Curate entity crops.
        selected_ids = await select_assets(
            prompt, entity_items, self.anthropic_client, max_items=max_items
        )
        selected_set = set(selected_ids)
        selected_items = [e for e in entity_items if e.item_id in selected_set]

        # Stage 6: Compose with the first available image as background.
        background = next((r.image for r in search_results if r.image is not None), None)
        collage = self.compose_fn(selected_items, canvas_size, background)

        # Stage 7: Save output.
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

    def _build_search_results_from_raw(self, raw_results: list[ImageResult]) -> list[SearchResult]:
        """Build SearchResult objects for all raw results.

        Args:
            raw_results: Raw image results from the search plan execution.

        Returns:
            Deduplicated list of SearchResult objects ready for thumbnail download.
        """
        seen_ids: set[str] = set()
        results: list[SearchResult] = []
        for item in raw_results:
            ident = item.get("identifier", "")
            if not ident or ident in seen_ids:
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
