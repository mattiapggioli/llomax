from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
from loguru import logger
from PIL import Image

from llomax.models import SourceImage


async def download_thumbnails(
    sources: list[SourceImage],
    cache_dir: Path = Path("output/thumbnails"),
) -> None:
    """Download thumbnail images to disk and set ``local_path`` on each source.

    Images are saved to ``cache_dir/{external_id}.jpg``. Sources without a
    ``thumbnail_url`` in their metadata are skipped. Failed downloads log a
    warning and leave ``local_path`` as ``None``. Already-cached files are
    reused without re-downloading.

    Args:
        sources: Source images to populate with downloaded files.
            ``local_path`` is set in place.
        cache_dir: Directory for cached thumbnail files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30) as client:
        for source in sources:
            thumbnail_url = source.metadata.get("thumbnail_url", "")
            if not thumbnail_url:
                continue

            local_path = cache_dir / f"{source.external_id}.jpg"
            if local_path.exists():
                source.local_path = local_path
                continue

            try:
                resp = await client.get(thumbnail_url)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                img.load()
                img.save(local_path)
                source.local_path = local_path
            except Exception:
                logger.warning("Failed to download thumbnail for %s", source.external_id)
