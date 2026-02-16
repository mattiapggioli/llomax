"""Parsing utilities for Internet Archive search tool responses."""

from __future__ import annotations

import json

from llomax.models import SearchResult


def parse_search_results(raw: str) -> list[SearchResult]:
    """Parse a JSON string from ``search_images_tool`` into model objects.

    Items missing an ``identifier`` field are silently skipped.
    Non-dict entries and non-list top-level values return an empty list.

    Args:
        raw: Raw JSON string returned by the MCP ``search_images_tool``.

    Returns:
        A list of ``SearchResult`` objects with ``image`` set to ``None``.
    """
    items = json.loads(raw)
    if not isinstance(items, list):
        return []
    return [
        SearchResult(
            identifier=item.get("identifier", ""),
            title=item.get("title", ""),
            thumbnail_url=item.get("thumbnail_url", ""),
            details_url=item.get("details_url", ""),
        )
        for item in items
        if isinstance(item, dict) and item.get("identifier")
    ]
