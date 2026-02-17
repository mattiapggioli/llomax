"""End-to-end collage pipeline orchestrating search, analysis, and composition."""

from __future__ import annotations

import os
from typing import Callable

import anthropic

from llomax.analysis.client import AnalysisClient
from llomax.composition.composer import compose as default_compose
from llomax.models import AnalysisResult, CollageOutput, SearchResult
from llomax.output import save_run
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
        self.search_agent = search_agent
        self.analysis_client = analysis_client
        self.anthropic_client = anthropic_client or search_agent.client
        self.compose_fn = compose_fn

    async def run(
        self,
        prompt: str,
        canvas_size: tuple[int, int] = (1024, 1024),
    ) -> CollageOutput:
        """Execute the full pipeline from prompt to collage."""

        raw_results = await self.search_agent.search(prompt)

        # 2. Dedup & sanitize
        seen: dict[str, dict] = {}
        for item in raw_results:
            ident = item.get("identifier", "")
            if ident and ident not in seen:
                seen[ident] = {
                    "identifier": ident,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "year": item.get("date", "")[:4] if item.get("date") else "",
                }
        sanitized = list(seen.values())

        # 3. Curation: select best assets
        selected_ids = await select_assets(prompt, sanitized, self.anthropic_client)

        # 4. Convert to SearchResult and download thumbnails
        selected_set = set(selected_ids)
        search_results = [
            SearchResult(
                identifier=item["identifier"],
                title=item.get("title", ""),
                thumbnail_url=f"https://archive.org/services/img/{item['identifier']}",
                details_url=f"https://archive.org/details/{item['identifier']}",
                description=item.get("description", ""),
                year=item.get("date", "")[:4] if item.get("date") else "",
            )
            for item in raw_results
            if item.get("identifier") in selected_set
        ]
        # Dedup (keep first occurrence)
        seen_ids: set[str] = set()
        deduped: list[SearchResult] = []
        for sr in search_results:
            if sr.identifier not in seen_ids:
                seen_ids.add(sr.identifier)
                deduped.append(sr)
        search_results = deduped

        await download_thumbnails(search_results)

        # 5. Analysis -> Composition (unchanged)
        elements = await self.analysis_client.analyze(search_results)
        collage = self.compose_fn(elements, canvas_size)

        output_dir = os.environ.get("OUTPUT_DIR", "output")
        save_run(collage, search_results, prompt, canvas_size, output_dir)

        return collage
