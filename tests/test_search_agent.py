from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from PIL import Image

from llomax.models import SourceImage
from llomax.search.clients.internet_archive_client import ImageResult, InternetArchiveClient
from llomax.search.curator import select_sources
from llomax.search.internet_archive_agent import MAX_AGENT_TURNS, InternetArchiveAgent
from llomax.search.thumbnails import download_thumbnails

# ---------------------------------------------------------------------------
# InternetArchiveClient tests
# ---------------------------------------------------------------------------


class TestInternetArchiveClient:
    def test_search_images_forces_mediatype(self):
        with patch("llomax.search.clients.internet_archive_client.internetarchive") as mock_ia:
            mock_ia.search_items.return_value = iter([])
            client = InternetArchiveClient()
            client.search_images(keywords="flowers")
            call_args = mock_ia.search_items.call_args
            assert "mediatype:image" in call_args[0][0]

    def test_search_images_with_collection(self):
        with patch("llomax.search.clients.internet_archive_client.internetarchive") as mock_ia:
            mock_ia.search_items.return_value = iter([])
            client = InternetArchiveClient()
            client.search_images(keywords="flowers", collection="nasa")
            query = mock_ia.search_items.call_args[0][0]
            assert "collection:nasa" in query

    def test_search_images_with_date_filter(self):
        with patch("llomax.search.clients.internet_archive_client.internetarchive") as mock_ia:
            mock_ia.search_items.return_value = iter([])
            client = InternetArchiveClient()
            client.search_images(keywords="flowers", date_filter="1900 TO 1950")
            query = mock_ia.search_items.call_args[0][0]
            assert "date:[1900 TO 1950]" in query

    def test_search_images_transforms_results(self):
        with patch("llomax.search.clients.internet_archive_client.internetarchive") as mock_ia:
            mock_ia.search_items.return_value = iter(
                [
                    {
                        "identifier": "img1",
                        "title": "Sunset",
                        "creator": "Author",
                        "date": "2020-01-01",
                        "description": "A sunset",
                    }
                ]
            )
            client = InternetArchiveClient()
            results = client.search_images(keywords="sunset")
            assert len(results) == 1
            assert results[0]["identifier"] == "img1"
            assert results[0]["thumbnail_url"] == "https://archive.org/services/img/img1"
            assert results[0]["details_url"] == "https://archive.org/details/img1"

    def test_search_images_skips_items_without_identifier(self):
        with patch("llomax.search.clients.internet_archive_client.internetarchive") as mock_ia:
            mock_ia.search_items.return_value = iter(
                [{"title": "No ID"}, {"identifier": "ok", "title": "Has ID"}]
            )
            client = InternetArchiveClient()
            results = client.search_images(keywords="test")
            assert len(results) == 1
            assert results[0]["identifier"] == "ok"

    def test_find_collections_forces_mediatype(self):
        with patch("llomax.search.clients.internet_archive_client.internetarchive") as mock_ia:
            mock_ia.search_items.return_value = iter([])
            client = InternetArchiveClient()
            client.find_collections(keywords="space")
            query = mock_ia.search_items.call_args[0][0]
            assert "mediatype:collection" in query

    def test_find_collections_returns_results(self):
        with patch("llomax.search.clients.internet_archive_client.internetarchive") as mock_ia:
            mock_ia.search_items.return_value = iter(
                [{"identifier": "nasa", "title": "NASA", "description": "NASA images"}]
            )
            client = InternetArchiveClient()
            results = client.find_collections(keywords="space")
            assert len(results) == 1
            assert results[0]["identifier"] == "nasa"

    def test_get_curated_collections(self):
        client = InternetArchiveClient()
        collections = client.get_curated_collections()
        assert len(collections) > 0
        identifiers = [c["identifier"] for c in collections]
        assert "nasa" in identifiers
        assert "smithsonian" in identifiers


# ---------------------------------------------------------------------------
# Tool dispatch tests
# ---------------------------------------------------------------------------


class TestDispatchTool:
    def _make_agent(self, ia_client: InternetArchiveClient) -> InternetArchiveAgent:
        return InternetArchiveAgent(anthropic_client=AsyncMock(), ia_client=ia_client)

    def test_dispatch_search_images(self):
        mock_client = MagicMock(spec=InternetArchiveClient)
        mock_client.search_images.return_value = [
            ImageResult(identifier="x", title="X", thumbnail_url="", details_url="")
        ]
        agent = self._make_agent(mock_client)
        result = agent._dispatch_tool("search_images", {"keywords": "test"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["identifier"] == "x"

    def test_dispatch_find_collections(self):
        mock_client = MagicMock(spec=InternetArchiveClient)
        mock_client.find_collections.return_value = []
        agent = self._make_agent(mock_client)
        result = agent._dispatch_tool("find_collections", {"keywords": "space"})
        assert json.loads(result) == []

    def test_dispatch_search_images_forwards_max_results(self):
        mock_client = MagicMock(spec=InternetArchiveClient)
        mock_client.search_images.return_value = []
        agent = self._make_agent(mock_client)
        agent._dispatch_tool("search_images", {"keywords": "test", "max_results": 50})
        mock_client.search_images.assert_called_once_with(
            keywords="test", collection=None, date_filter=None, max_results=50
        )

    def test_dispatch_unknown_tool(self):
        mock_client = MagicMock(spec=InternetArchiveClient)
        agent = self._make_agent(mock_client)
        result = agent._dispatch_tool("unknown_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed


# ---------------------------------------------------------------------------
# InternetArchiveAgent tests
# ---------------------------------------------------------------------------


def _make_tool_use_response(tool_calls: list[dict], stop_reason="tool_use"):
    blocks = []
    for tc in tool_calls:
        block = MagicMock()
        block.type = "tool_use"
        block.id = tc["id"]
        block.name = tc["name"]
        block.input = tc.get("input", {})
        blocks.append(block)
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    return resp


def _make_end_turn_response():
    resp = MagicMock()
    resp.content = [MagicMock(type="text", text="Done searching.")]
    resp.stop_reason = "end_turn"
    return resp


class TestInternetArchiveAgent:
    async def test_single_search_then_end(self):
        mock_ia = MagicMock(spec=InternetArchiveClient)
        mock_ia.search_images.return_value = [
            ImageResult(
                identifier="r1",
                title="Result",
                thumbnail_url="https://archive.org/services/img/r1",
                details_url="https://archive.org/details/r1",
            )
        ]

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _make_tool_use_response(
                    [{"id": "t1", "name": "search_images", "input": {"keywords": "test"}}]
                ),
                _make_end_turn_response(),
            ]
        )

        agent = InternetArchiveAgent(anthropic_client=mock_anthropic, ia_client=mock_ia)
        results = await agent.search("test prompt", max_items=10)

        assert len(results) == 1
        assert results[0]["identifier"] == "r1"
        assert mock_anthropic.messages.create.call_count == 2
        first_call_messages = mock_anthropic.messages.create.call_args_list[0].kwargs["messages"]
        assert "The user wants 10 images" in first_call_messages[0]["content"]

    async def test_deduplication(self):
        mock_ia = MagicMock(spec=InternetArchiveClient)
        mock_ia.search_images.side_effect = [
            [
                ImageResult(identifier="dup", title="First", thumbnail_url="", details_url=""),
                ImageResult(identifier="u1", title="U1", thumbnail_url="", details_url=""),
            ],
            [
                ImageResult(identifier="dup", title="Second", thumbnail_url="", details_url=""),
                ImageResult(identifier="u2", title="U2", thumbnail_url="", details_url=""),
            ],
        ]

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _make_tool_use_response(
                    [{"id": "t1", "name": "search_images", "input": {"keywords": "q1"}}]
                ),
                _make_tool_use_response(
                    [{"id": "t2", "name": "search_images", "input": {"keywords": "q2"}}]
                ),
                _make_end_turn_response(),
            ]
        )

        agent = InternetArchiveAgent(anthropic_client=mock_anthropic, ia_client=mock_ia)
        results = await agent.search("test")

        identifiers = [r["identifier"] for r in results]
        assert len(identifiers) == 3
        assert identifiers.count("dup") == 1
        dup_result = next(r for r in results if r["identifier"] == "dup")
        assert dup_result["title"] == "First"

    async def test_max_turns_safety(self):
        mock_ia = MagicMock(spec=InternetArchiveClient)
        mock_ia.search_images.return_value = [
            ImageResult(identifier="x", title="X", thumbnail_url="", details_url="")
        ]

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                [{"id": "t1", "name": "search_images", "input": {"keywords": "loop"}}]
            )
        )

        agent = InternetArchiveAgent(anthropic_client=mock_anthropic, ia_client=mock_ia)
        results = await agent.search("infinite loop")

        assert mock_anthropic.messages.create.call_count == MAX_AGENT_TURNS
        assert len(results) >= 1

    async def test_end_turn_on_first_response(self):
        mock_ia = MagicMock(spec=InternetArchiveClient)
        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(return_value=_make_end_turn_response())

        agent = InternetArchiveAgent(anthropic_client=mock_anthropic, ia_client=mock_ia)
        results = await agent.search("nothing")

        assert results == []
        assert mock_anthropic.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Curator tests
# ---------------------------------------------------------------------------


def _make_source(external_id: str, title: str = "") -> SourceImage:
    return SourceImage(
        external_id=external_id,
        title=title or external_id.upper(),
        description="",
        local_path=None,
        metadata={"year": "1900", "creator": "", "thumbnail_url": "", "details_url": ""},
    )


class TestCurator:
    async def test_selects_identifiers(self):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '["id1", "id3"]'
        mock_response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        candidates = [_make_source("id1"), _make_source("id2"), _make_source("id3")]
        selected = await select_sources("prompt", candidates, {}, mock_client)
        assert selected == ["id1", "id3"]

    async def test_handles_markdown_fenced_json(self):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '```json\n["id1"]\n```'
        mock_response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        selected = await select_sources("prompt", [], {}, mock_client)
        assert selected == ["id1"]

    async def test_handles_non_list_response(self):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"not": "a list"}'
        mock_response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        selected = await select_sources("prompt", [], {}, mock_client)
        assert selected == []

    async def test_respects_max_fragments(self):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '["id1", "id2"]'
        mock_response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        candidates = [_make_source("id1"), _make_source("id2")]
        selected = await select_sources("prompt", candidates, {}, mock_client, max_fragments=5)
        assert selected == ["id1", "id2"]
        user_msg = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "5" in user_msg

    async def test_filters_non_string_items(self):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '["id1", 42, "id2", null]'
        mock_response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        selected = await select_sources("prompt", [], {}, mock_client)
        assert selected == ["id1", "id2"]

    async def test_candidate_summaries_sent_to_llm(self):
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"
        mock_response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        candidates = [_make_source("img1", title="A Fine Painting")]
        await select_sources("prompt", candidates, {}, mock_client)

        user_msg = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "img1" in user_msg
        assert "A Fine Painting" in user_msg
        assert "fragment_count" in user_msg

    async def test_fragment_labels_included_in_summary(self):
        from PIL import Image as PILImage

        from llomax.models import Fragment

        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"
        mock_response.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        fragment = Fragment(
            source_id="img1",
            image_rgba=PILImage.new("RGBA", (10, 10)),
            bounding_box=(0, 0, 10, 10),
            label="person",
        )
        candidates = [_make_source("img1")]
        await select_sources("prompt", candidates, {"img1": [fragment]}, mock_client)

        user_msg = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "person" in user_msg
        assert "fragment_count" in user_msg


# ---------------------------------------------------------------------------
# Download thumbnails tests
# ---------------------------------------------------------------------------


class TestDownloadThumbnails:
    async def test_downloads_images(self, tmp_path: Path):
        img = Image.new("RGB", (10, 10), "blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        sources = [
            SourceImage(
                external_id="img1",
                title="Test",
                description="",
                local_path=None,
                metadata={"thumbnail_url": "https://archive.org/services/img/img1"},
            )
        ]

        with patch("llomax.search.thumbnails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await download_thumbnails(sources, cache_dir=tmp_path)

        assert sources[0].local_path is not None
        assert sources[0].local_path.exists()

    async def test_failed_download_leaves_local_path_none(self, tmp_path: Path):
        sources = [
            SourceImage(
                external_id="bad",
                title="Bad",
                description="",
                local_path=None,
                metadata={"thumbnail_url": "https://archive.org/services/img/bad"},
            )
        ]

        with patch("llomax.search.thumbnails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("fail"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await download_thumbnails(sources, cache_dir=tmp_path)

        assert sources[0].local_path is None

    async def test_skips_sources_without_thumbnail_url(self, tmp_path: Path):
        sources = [
            SourceImage(
                external_id="no_url",
                title="No URL",
                description="",
                local_path=None,
                metadata={},
            )
        ]
        await download_thumbnails(sources, cache_dir=tmp_path)
        assert sources[0].local_path is None

    async def test_reuses_cached_file(self, tmp_path: Path):
        # Pre-create the cached file
        cached = tmp_path / "img1.jpg"
        Image.new("RGB", (10, 10), "red").save(cached)

        sources = [
            SourceImage(
                external_id="img1",
                title="Cached",
                description="",
                local_path=None,
                metadata={"thumbnail_url": "https://archive.org/services/img/img1"},
            )
        ]

        with patch("llomax.search.thumbnails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            await download_thumbnails(sources, cache_dir=tmp_path)

            # File already cached — no HTTP request should be made
            mock_client.get.assert_not_called()

        assert sources[0].local_path == cached
