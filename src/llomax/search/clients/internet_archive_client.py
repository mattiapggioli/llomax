from __future__ import annotations

from typing import Required, TypedDict

import internetarchive

THUMBNAIL_URL_TEMPLATE = "https://archive.org/services/img/{identifier}"
DETAILS_URL_TEMPLATE = "https://archive.org/details/{identifier}"

IMAGE_FIELDS = ["identifier", "title", "creator", "date", "description"]
COLLECTION_FIELDS = ["identifier", "title", "description"]

CURATED_COLLECTIONS = [
    {"identifier": "nasa", "title": "NASA Images", "description": "NASA's image archive"},
    {
        "identifier": "flickrcommons",
        "title": "Flickr Commons",
        "description": "The commons on Flickr",
    },
    {
        "identifier": "smithsonian",
        "title": "Smithsonian",
        "description": "Smithsonian Institution collections",
    },
    {
        "identifier": "brooklynmuseum",
        "title": "Brooklyn Museum",
        "description": "Brooklyn Museum image collection",
    },
    {
        "identifier": "library_of_congress",
        "title": "Library of Congress",
        "description": "Library of Congress digital collections",
    },
    {
        "identifier": "biodiversity",
        "title": "Biodiversity Heritage Library",
        "description": "Biodiversity Heritage Library images",
    },
    {
        "identifier": "metropolitanmuseumofart-gallery",
        "title": "Metropolitan Museum of Art",
        "description": "The Met's open access images",
    },
    {
        "identifier": "coverartarchive",
        "title": "Cover Art Archive",
        "description": "Music cover art",
    },
]


class ImageResult(TypedDict, total=False):
    """A single image result from an Internet Archive search."""

    identifier: Required[str]
    title: str
    creator: str
    date: str
    description: str
    thumbnail_url: str
    details_url: str


class CollectionResult(TypedDict, total=False):
    """A single collection result from an Internet Archive search."""

    identifier: Required[str]
    title: str
    description: str
    details_url: str


class CuratedCollection(TypedDict):
    """A hardcoded curated Internet Archive collection."""

    identifier: str
    title: str
    description: str


class IAClient:
    """Synchronous client for Internet Archive searches."""

    def search_images(
        self,
        keywords: str,
        collection: str | None = None,
        date_filter: str | None = None,
        max_results: int = 20,
    ) -> list[ImageResult]:
        """Search for images using Lucene keywords, with optional collection and date filters."""
        query = self._build_query(keywords, "image", collection, date_filter)
        results: list[ImageResult] = []
        for item in internetarchive.search_items(
            query, fields=IMAGE_FIELDS, max_results=max_results
        ):
            identifier = item.get("identifier", "")
            if not identifier:
                continue
            results.append(
                ImageResult(
                    identifier=identifier,
                    title=item.get("title", ""),
                    creator=item.get("creator", ""),
                    date=item.get("date", ""),
                    description=item.get("description", ""),
                    thumbnail_url=THUMBNAIL_URL_TEMPLATE.format(identifier=identifier),
                    details_url=DETAILS_URL_TEMPLATE.format(identifier=identifier),
                )
            )
        return results

    def find_collections(
        self,
        keywords: str,
        max_results: int = 10,
    ) -> list[CollectionResult]:
        """Search for Internet Archive collections by keyword."""
        query = self._build_query(keywords, "collection")
        results: list[CollectionResult] = []
        for item in internetarchive.search_items(
            query, fields=COLLECTION_FIELDS, max_results=max_results
        ):
            identifier = item.get("identifier", "")
            if not identifier:
                continue
            results.append(
                CollectionResult(
                    identifier=identifier,
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    details_url=DETAILS_URL_TEMPLATE.format(identifier=identifier),
                )
            )
        return results

    def get_curated_collections(self) -> list[CuratedCollection]:
        """Return the hardcoded list of curated collections."""
        return list(CURATED_COLLECTIONS)

    def _build_query(
        self,
        keywords: str,
        mediatype: str,
        collection: str | None = None,
        date_filter: str | None = None,
    ) -> str:
        """Build a Lucene query string with mediatype and optional filters.

        Args:
            keywords: Search keywords (Lucene boolean syntax).
            mediatype: Required mediatype filter (e.g. "image", "collection").
            collection: Optional collection identifier to filter by.
            date_filter: Optional date range (e.g. "1900 TO 1950").

        Returns:
            Formatted Lucene query string.
        """
        query = f"({keywords}) AND mediatype:{mediatype}"
        if collection:
            query += f" AND collection:{collection}"
        if date_filter:
            query += f" AND date:[{date_filter}]"
        return query
