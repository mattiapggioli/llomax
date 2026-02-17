"""Internet Archive client wrapping the ``internetarchive`` library."""

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
    identifier: Required[str]
    title: str
    creator: str
    date: str
    description: str
    thumbnail_url: str
    details_url: str


class CollectionResult(TypedDict, total=False):
    identifier: Required[str]
    title: str
    description: str
    details_url: str


class CuratedCollection(TypedDict):
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
        query = f"({keywords}) AND mediatype:image"
        if collection:
            query += f" AND collection:{collection}"
        if date_filter:
            query += f" AND date:[{date_filter}]"

        results: list[ImageResult] = []
        search = internetarchive.search_items(query, fields=IMAGE_FIELDS, max_results=max_results)
        for item in search:
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
        query = f"({keywords}) AND mediatype:collection"

        results: list[CollectionResult] = []
        search = internetarchive.search_items(
            query, fields=COLLECTION_FIELDS, max_results=max_results
        )
        for item in search:
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
        return list(CURATED_COLLECTIONS)
