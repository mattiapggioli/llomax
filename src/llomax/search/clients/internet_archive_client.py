from __future__ import annotations

import itertools
import urllib.parse
from typing import Required, TypedDict

import internetarchive
from loguru import logger

THUMBNAIL_URL_TEMPLATE = "https://archive.org/services/img/{identifier}"
DETAILS_URL_TEMPLATE = "https://archive.org/details/{identifier}"

IMAGE_FIELDS = ["identifier", "title", "creator", "date", "description"]
COLLECTION_FIELDS = ["identifier", "title", "description"]

CURATED_COLLECTIONS: list[CuratedCollection] = [
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

# Terms that are implicit when searching within a specific curated collection.
# These are stripped from the keyword list to avoid over-constraining queries.
_COLLECTION_IMPLICIT_TERMS: dict[str, set[str]] = {
    "nasa": {"nasa", "space", "astronaut", "astronauts", "rocket", "spacecraft", "satellite"},
    "flickrcommons": {"flickr"},
    "smithsonian": {"smithsonian"},
    "brooklynmuseum": {"brooklyn", "museum"},
    "library_of_congress": {"library", "congress", "loc"},
    "biodiversity": {"biodiversity", "biology", "biological"},
    "metropolitanmuseumofart-gallery": {"metropolitan", "museum", "met"},
    "coverartarchive": {"cover", "album"},
}


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


class InternetArchiveClient:
    """Synchronous client for Internet Archive searches."""

    def search_images(
        self,
        keywords: list[str],
        collection: str | None = None,
        date_filter: str | None = None,
        max_results: int = 20,
    ) -> list[ImageResult]:
        """Search for images by keyword list, with optional collection and date filters.

        Args:
            keywords: List of search terms joined with OR. Terms implicit to the
                collection (e.g. "space" for the nasa collection) are stripped
                automatically to avoid over-constraining the query.
            collection: Optional Internet Archive collection identifier to restrict results.
            date_filter: Optional date range in IA Lucene format, e.g. ``"1900 TO 1950"``.
            max_results: Maximum number of results to return.

        Returns:
            List of ``ImageResult`` dicts with identifier, title, creator, date,
            description, thumbnail_url, and details_url.
        """
        query = self._build_query(keywords, "image", collection, date_filter)
        logger.debug(
            "[IA] search_images query: {}  url: https://archive.org/advancedsearch.php?q={}",
            query,
            urllib.parse.quote(query),
        )
        items = itertools.islice(
            internetarchive.search_items(query, fields=IMAGE_FIELDS), max_results
        )
        return [
            self._image_result_from_item(item)
            for item in items
            if item.get("identifier", "")
        ]

    def find_collections(
        self,
        keywords: list[str],
        max_results: int = 10,
    ) -> list[CollectionResult]:
        """Search for Internet Archive collections by keyword.

        Args:
            keywords: List of search terms joined with OR to match collection
                titles and descriptions.
            max_results: Maximum number of results to return.

        Returns:
            List of ``CollectionResult`` dicts with identifier, title,
            description, and details_url.
        """
        query = self._build_query(keywords, "collection")
        logger.debug(
            "[IA] find_collections query: {}  url: https://archive.org/advancedsearch.php?q={}",
            query,
            urllib.parse.quote(query),
        )
        items = itertools.islice(
            internetarchive.search_items(query, fields=COLLECTION_FIELDS), max_results
        )
        return [
            self._collection_result_from_item(item)
            for item in items
            if item.get("identifier", "")
        ]

    def get_curated_collections(self) -> list[CuratedCollection]:
        """Return the hardcoded list of curated collections."""
        return list(CURATED_COLLECTIONS)

    def _image_result_from_item(self, item: dict) -> ImageResult:
        """Build an ``ImageResult`` from a raw Internet Archive search item.

        Args:
            item: Raw search result dict from the ``internetarchive`` library.

        Returns:
            ``ImageResult`` with identifier, title, creator, date, description,
            thumbnail_url, and details_url populated.
        """
        identifier = item["identifier"]
        return ImageResult(
            identifier=identifier,
            title=item.get("title", ""),
            creator=item.get("creator", ""),
            date=item.get("date", ""),
            description=item.get("description", ""),
            thumbnail_url=THUMBNAIL_URL_TEMPLATE.format(identifier=identifier),
            details_url=DETAILS_URL_TEMPLATE.format(identifier=identifier),
        )

    def _collection_result_from_item(self, item: dict) -> CollectionResult:
        """Build a ``CollectionResult`` from a raw Internet Archive search item.

        Args:
            item: Raw search result dict from the ``internetarchive`` library.

        Returns:
            ``CollectionResult`` with identifier, title, description,
            and details_url populated.
        """
        identifier = item["identifier"]
        return CollectionResult(
            identifier=identifier,
            title=item.get("title", ""),
            description=item.get("description", ""),
            details_url=DETAILS_URL_TEMPLATE.format(identifier=identifier),
        )

    def _build_query(
        self,
        keywords: list[str],
        mediatype: str,
        collection: str | None = None,
        date_filter: str | None = None,
        operator: str = "OR",
    ) -> str:
        """Build a Lucene query string with mediatype and optional filters.

        Keywords are joined with ``operator`` (default OR). When ``collection``
        matches a curated collection, terms implicit to that collection are
        stripped before joining to avoid over-constraining the query; the
        original list is used as a fallback if all terms would be stripped.

        Args:
            keywords: List of search terms.
            mediatype: Required mediatype filter (e.g. "image", "collection").
            collection: Optional collection identifier to filter by.
            date_filter: Optional date range (e.g. "1900 TO 1950").
            operator: Boolean operator used to join keywords (default ``"OR"``).

        Returns:
            Formatted Lucene query string.
        """
        effective = keywords
        if collection and collection in _COLLECTION_IMPLICIT_TERMS:
            implicit = _COLLECTION_IMPLICIT_TERMS[collection]
            cleaned = [k for k in keywords if k.lower() not in implicit]
            if cleaned:
                effective = cleaned

        joined = f" {operator} ".join(effective)
        query = f"({joined}) AND mediatype:{mediatype}"
        if collection:
            query += f" AND collection:{collection}"
        if date_filter:
            query += f" AND date:[{date_filter}]"
        return query
