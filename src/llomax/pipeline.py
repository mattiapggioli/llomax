from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import anthropic

from llomax.analysis.annotator import PlaceholderAnnotator
from llomax.analysis.client import AnalysisClient
from llomax.composition.composer import compose as default_compose
from llomax.models import CollageOutput, Fragment, SourceImage
from llomax.output import save_run
from llomax.search.clients.internet_archive_client import ImageResult
from llomax.search.curator import select_sources
from llomax.search.internet_archive_agent import InternetArchiveAgent
from llomax.search.thumbnails import download_thumbnails


class Pipeline:
    """Multi-stage pipeline: discovery → source selection → segmentation → annotation → composition."""

    def __init__(
        self,
        search_agent: InternetArchiveAgent,
        analysis_client: AnalysisClient,
        annotator: PlaceholderAnnotator | None = None,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        thumbnails_dir: Path | str = Path("output/thumbnails"),
        compose_fn: Callable[
            [list[Fragment], tuple[int, int]], CollageOutput
        ] = default_compose,
    ) -> None:
        """Initialize the pipeline.

        Args:
            search_agent: Agent for discovering images on the Internet Archive.
            analysis_client: Backend for segmenting source images into fragments.
                Use ``Segmenter`` for SAM-based segmentation or
                ``PlaceholderAnalysisClient`` for testing without a model.
            annotator: Annotation backend that labels fragments. Defaults to
                ``PlaceholderAnnotator`` when not provided.
            anthropic_client: Anthropic async client for the source-selection
                stage. Defaults to the search agent's client.
            thumbnails_dir: Directory for cached thumbnail files.
            compose_fn: Callable that arranges fragments onto a canvas.
        """
        self.search_agent = search_agent
        self.analysis_client = analysis_client
        self.annotator = annotator or PlaceholderAnnotator()
        self.anthropic_client = anthropic_client or search_agent.client
        self.thumbnails_dir = Path(thumbnails_dir)
        self.compose_fn = compose_fn

    async def run(
        self,
        prompt: str,
        canvas_size: tuple[int, int] = (1024, 1024),
        max_items: int = 20,
    ) -> CollageOutput:
        """Execute the full pipeline from prompt to collage.

        Pipeline stages:
        1. ``plan_search`` — LLM registers search intents without seeing raw data.
        2. ``_execute_search_plan`` — Python executes each query directly.
        3. ``download_thumbnails`` — Fetch and cache source image files.
        4. ``select_sources`` — LLM curator picks the best source images.
        5. ``analysis_client.analyze`` — Segment selected images into fragments.
        6. ``annotator.annotate`` — Populate placeholder labels and descriptions.
        7. ``compose_fn`` — Place fragments onto the canvas.
        8. ``save_run`` — Persist collage and provenance metadata.

        Args:
            prompt: Creative text prompt describing the desired collage.
            canvas_size: ``(width, height)`` in pixels for the output canvas.
            max_items: Target number of source images for curation.

        Returns:
            The composed ``CollageOutput``.
        """
        # Stage 1: Plan searches.
        search_plan = await self.search_agent.plan_search(prompt, max_items=max_items)

        # Stage 2: Execute plan directly in Python.
        raw_results = self._execute_search_plan(search_plan)
        source_candidates = self._build_source_images(raw_results)

        # Stage 3: Download thumbnails for all candidates.
        await download_thumbnails(source_candidates, self.thumbnails_dir)

        # Stage 4: Select the best source images.
        selected_ids = await select_sources(
            prompt, source_candidates, self.anthropic_client, max_sources=max_items
        )
        selected_set = set(selected_ids)
        selected_sources = [s for s in source_candidates if s.external_id in selected_set]

        # Stage 5: Segment selected sources into fragments.
        fragments: list[Fragment] = await self.analysis_client.analyze(selected_sources)

        # Stage 6: Annotate fragments with placeholder labels and descriptions.
        self.annotator.annotate(selected_sources, fragments)

        # Stage 7: Compose collage.
        collage = self.compose_fn(fragments, canvas_size)

        # Stage 8: Save output.
        output_dir = os.environ.get("OUTPUT_DIR", "output")
        save_run(collage, source_candidates, prompt, canvas_size, output_dir)

        return collage

    def _execute_search_plan(self, plan: list[dict]) -> list[ImageResult]:
        """Execute each item in the search plan and return deduplicated results.

        Args:
            plan: List of search plan items from ``plan_search``.

        Returns:
            Deduplicated list of ``ImageResult`` items, keyed by identifier.
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

    def _build_source_images(self, raw_results: list[ImageResult]) -> list[SourceImage]:
        """Build ``SourceImage`` objects from raw Internet Archive results.

        Args:
            raw_results: Raw image results from the search plan execution.

        Returns:
            Deduplicated list of ``SourceImage`` objects ready for thumbnail download.
        """
        seen_ids: set[str] = set()
        sources: list[SourceImage] = []
        for item in raw_results:
            ident = item.get("identifier", "")
            if not ident or ident in seen_ids:
                continue
            seen_ids.add(ident)
            sources.append(
                SourceImage(
                    external_id=ident,
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    local_path=None,
                    metadata={
                        "creator": item.get("creator", ""),
                        "year": item.get("date", "")[:4] if item.get("date") else "",
                        "thumbnail_url": f"https://archive.org/services/img/{ident}",
                        "details_url": f"https://archive.org/details/{ident}",
                    },
                )
            )
        return sources
