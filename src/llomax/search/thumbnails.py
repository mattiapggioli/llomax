"""Async thumbnail downloader for Internet Archive search results."""

from __future__ import annotations

import logging
from io import BytesIO

import httpx
from PIL import Image

from llomax.models import SearchResult

logger = logging.getLogger(__name__)


async def download_thumbnails(results: list[SearchResult]) -> None:
    """Download thumbnail images into each result's ``image`` field.

    Results without a ``thumbnail_url`` are skipped. Failed downloads
    log a warning and leave ``image`` as ``None``.

    Args:
        results: Search results to populate with downloaded images.
            Modified in place.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        for result in results:
            if not result.thumbnail_url:
                continue
            try:
                resp = await client.get(result.thumbnail_url)
                resp.raise_for_status()
                result.image = Image.open(BytesIO(resp.content))
                result.image.load()
            except Exception:
                logger.warning("Failed to download thumbnail for %s", result.identifier)
