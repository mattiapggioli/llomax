from __future__ import annotations

import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable

import anthropic
from loguru import logger
from PIL import Image

from llomax.analysis.annotator import PlaceholderAnnotator
from llomax.analysis.client import AnalysisClient
from llomax.composition.composer import compose as default_compose
from llomax.core.hooks import HookManager, PipelineState
from llomax.models import CollageOutput, Fragment, SourceImage
from llomax.output import save_run
from llomax.search.clients.internet_archive_client import ImageResult
from llomax.search.curator import select_fragments
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
            [list[Fragment], tuple[int, int], Image.Image | None], CollageOutput
        ] = default_compose,
        hooks: HookManager | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            search_agent: Agent for discovering images on the Internet Archive.
            analysis_client: Backend for segmenting source images into fragments.
                Use ``Segmenter`` for SAM-based segmentation,
                ``YoloAnalysisClient`` for YOLO instance segmentation, or
                ``PlaceholderAnalysisClient`` for testing without a model.
            annotator: Annotation backend that labels fragments. Defaults to
                ``PlaceholderAnnotator`` when not provided.
            anthropic_client: Anthropic async client for the source-selection
                stage. Defaults to the search agent's client.
            thumbnails_dir: Directory for cached thumbnail files.
            compose_fn: Callable that arranges fragments onto a canvas with an
                optional background image.
            hooks: Hook manager for registering ``after_curation``,
                ``pre_composition``, and ``composition_strategy`` hooks.
                A default empty manager is used when not provided.
        """
        self.search_agent = search_agent
        self.analysis_client = analysis_client
        self.annotator = annotator or PlaceholderAnnotator()
        self.anthropic_client = anthropic_client or search_agent.client
        self.thumbnails_dir = Path(thumbnails_dir)
        self.compose_fn = compose_fn
        self.hooks = hooks or HookManager()

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
        output_dir = Path(os.environ.get("OUTPUT_DIR", "output"))
        run_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        log_sink_id = logger.add(
            run_dir / "pipeline.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
            level="DEBUG",
            encoding="utf-8",
        )

        try:
            return await self._run_stages(prompt, canvas_size, max_items, output_dir, run_dir)
        finally:
            logger.remove(log_sink_id)

    async def _run_stages(
        self,
        prompt: str,
        canvas_size: tuple[int, int],
        max_items: int,
        output_dir: Path,
        run_dir: Path,
    ) -> CollageOutput:
        """Execute all pipeline stages with detailed logging.

        Args:
            prompt: Creative text prompt describing the desired collage.
            canvas_size: ``(width, height)`` in pixels for the output canvas.
            max_items: Target number of source images for curation.
            output_dir: Base directory for pipeline outputs.
            run_dir: Pre-created timestamped run directory.

        Returns:
            The composed ``CollageOutput``.
        """
        # Stage 1: Plan searches.
        logger.info("Stage 1 — Planning search. Prompt: {!r}", prompt)
        search_plan = await self.search_agent.plan_search(prompt, max_items=max_items)
        logger.info("Stage 1 complete — {} search intent(s) registered.", len(search_plan))
        for i, item in enumerate(search_plan, 1):
            logger.debug(
                "  Plan {}: keywords={!r}, collection={}, date_filter={}",
                i,
                item.get("keywords"),
                item.get("collection"),
                item.get("date_filter"),
            )

        # Stage 2: Execute plan directly in Python.
        logger.info("Stage 2 — Executing search plan ({} queries)...", len(search_plan))
        raw_results = self._execute_search_plan(search_plan)
        source_candidates = self._build_source_images(raw_results)
        logger.info(
            "Stage 2 complete — {} unique candidate source(s) discovered.", len(source_candidates)
        )

        # Stage 3: Download thumbnails for all candidates.
        logger.info(
            "Stage 3 — Downloading thumbnails for {} candidate(s) to {}...",
            len(source_candidates),
            self.thumbnails_dir,
        )
        await download_thumbnails(source_candidates, self.thumbnails_dir)
        cached = sum(1 for s in source_candidates if s.local_path is not None)
        logger.info(
            "Stage 3 complete — {}/{} thumbnail(s) available.", cached, len(source_candidates)
        )

        # Stage 4: Segment all candidates to discover their visual content.
        logger.info("Stage 4 — Segmenting {} candidate(s)...", len(source_candidates))
        all_fragments: list[Fragment] = await self.analysis_client.analyze(source_candidates)
        fragments_by_source: dict[str, list[Fragment]] = {}
        for f in all_fragments:
            fragments_by_source.setdefault(f.source_id, []).append(f)
        sources_with_fragments = sum(1 for v in fragments_by_source.values() if v)
        logger.info(
            "Stage 4 complete — {} fragment(s) extracted from {}/{} candidate(s).",
            len(all_fragments),
            sources_with_fragments,
            len(source_candidates),
        )
        label_counts = Counter(f.label for f in all_fragments)
        for label, count in label_counts.most_common():
            logger.debug("  {!r}: {} fragment(s)", label, count)

        # Stage 5: Curator picks individual fragments from the full pool.
        logger.info(
            "Stage 5 — Curating: selecting ~{} fragment(s) from {} available...",
            max_items,
            len(all_fragments),
        )
        selected_fragment_ids = await select_fragments(
            prompt,
            source_candidates,
            all_fragments,
            self.anthropic_client,
            max_fragments=max_items,
        )
        selected_id_set = set(selected_fragment_ids)
        fragments: list[Fragment] = [f for f in all_fragments if f.fragment_id in selected_id_set]
        selected_source_ids = {f.source_id for f in fragments}
        selected_sources = [s for s in source_candidates if s.external_id in selected_source_ids]
        logger.info(
            "Stage 5 complete — {} fragment(s) selected from {} source(s).",
            len(fragments),
            len(selected_sources),
        )
        source_frag_counts = Counter(f.source_id for f in fragments)
        for src in selected_sources:
            n = source_frag_counts[src.external_id]
            logger.debug("  {} fragment(s) from {} — {!r}", n, src.external_id, src.title)

        # Build shared pipeline state for hook points.
        state = PipelineState(
            prompt=prompt,
            canvas_size=canvas_size,
            sources=selected_sources,
            fragments=fragments,
        )

        # Hook: after_curation — hooks may flag a background source.
        await self.hooks.run("after_curation", state)
        if state.background_source_id:
            bg_src = next(
                (s for s in state.sources if s.external_id == state.background_source_id),
                None,
            )
            if bg_src and (bg_img := bg_src.load_image()):
                state.background_image = bg_img
                logger.info(
                    "Hooks — background set to {!r} ({}).",
                    bg_src.title,
                    state.background_source_id,
                )

        # Stage 6: Annotate fragments with placeholder labels and descriptions.
        logger.info("Stage 6 — Annotating {} fragment(s)...", len(state.fragments))
        self.annotator.annotate(state.sources, state.fragments)
        logger.info("Stage 6 complete — fragments annotated.")

        # Hook: pre_composition — hooks may apply palette or other transforms.
        await self.hooks.run("pre_composition", state)

        # Stage 7: Compose collage — use composition_strategy override if registered.
        logger.info(
            "Stage 7 — Composing collage: {} fragment(s) onto {}×{} canvas...",
            len(state.fragments),
            canvas_size[0],
            canvas_size[1],
        )
        composition_override = self.hooks.get_override("composition_strategy")
        if composition_override is not None:
            logger.info("Hooks — using composition_strategy override.")
            collage = await composition_override(state)
        else:
            collage = self.compose_fn(state.fragments, canvas_size, state.background_image)
        logger.info(
            "Stage 7 complete — collage composed ({} fragment(s) placed).",
            len(collage.fragment_provenance),
        )

        # Stage 8: Save output.
        logger.info("Stage 8 — Saving output to {}...", run_dir)
        save_run(collage, source_candidates, prompt, canvas_size, output_dir, run_dir=run_dir)
        logger.info("Pipeline complete. Run artifacts saved to {}", run_dir)

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

        Results are already deduplicated by ``_execute_search_plan``; items
        without an identifier are silently skipped.

        Args:
            raw_results: Raw image results from the search plan execution.

        Returns:
            List of ``SourceImage`` objects ready for thumbnail download.
        """
        return [
            self._source_image_from_item(item)
            for item in raw_results
            if item.get("identifier", "")
        ]

    def _source_image_from_item(self, item: ImageResult) -> SourceImage:
        """Build a ``SourceImage`` from a raw Internet Archive result.

        Args:
            item: Raw image result with at minimum an ``identifier`` key.

        Returns:
            ``SourceImage`` with metadata populated from the result fields.
        """
        ident = item["identifier"]
        return SourceImage(
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
