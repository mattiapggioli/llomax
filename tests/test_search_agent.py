from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from PIL import Image

from mcp.types import TextContent

from llomax.models import SearchResult
from llomax.search.agent import MAX_AGENT_TURNS, SearchAgent, run_agent_loop
from llomax.search.mcp import forward_tool_calls, mcp_tools_to_anthropic
from llomax.search.parsing import parse_search_results
from llomax.search.thumbnails import download_thumbnails


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestParseSearchResults:
    def test_basic_parsing(self):
        raw = json.dumps(
            [
                {
                    "identifier": "img1",
                    "title": "Sunset",
                    "thumbnail_url": "https://archive.org/services/img/img1",
                    "details_url": "https://archive.org/details/img1",
                },
                {
                    "identifier": "img2",
                    "title": "Mountains",
                    "thumbnail_url": "https://archive.org/services/img/img2",
                    "details_url": "https://archive.org/details/img2",
                },
            ]
        )
        results = parse_search_results(raw)
        assert len(results) == 2
        assert results[0].identifier == "img1"
        assert results[1].title == "Mountains"

    def test_empty_list(self):
        assert parse_search_results("[]") == []

    def test_missing_fields_default_to_empty_string(self):
        raw = json.dumps([{"identifier": "x"}])
        results = parse_search_results(raw)
        assert len(results) == 1
        assert results[0].title == ""
        assert results[0].thumbnail_url == ""
        assert results[0].details_url == ""

    def test_items_without_identifier_are_skipped(self):
        raw = json.dumps([{"title": "No ID"}, {"identifier": "ok", "title": "Has ID"}])
        results = parse_search_results(raw)
        assert len(results) == 1
        assert results[0].identifier == "ok"

    def test_non_list_returns_empty(self):
        assert parse_search_results('{"not": "a list"}') == []

    def test_non_dict_items_are_skipped(self):
        raw = json.dumps(["just_a_string", {"identifier": "ok"}])
        results = parse_search_results(raw)
        assert len(results) == 1


class TestMcpToolsToAnthropic:
    def test_converts_tools(self):
        @dataclass
        class FakeTool:
            name: str
            description: str | None
            inputSchema: dict

        tools = [
            FakeTool(
                name="search_images_tool",
                description="Search images",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            FakeTool(
                name="list_curated_collections_tool",
                description=None,
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        result = mcp_tools_to_anthropic(tools)
        assert len(result) == 2
        assert result[0] == {
            "name": "search_images_tool",
            "description": "Search images",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
        assert result[1]["description"] == ""


# ---------------------------------------------------------------------------
# Download thumbnails tests
# ---------------------------------------------------------------------------


class TestDownloadThumbnails:
    async def test_downloads_images(self):
        img = Image.new("RGB", (10, 10), "blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = png_bytes
        mock_response.raise_for_status = MagicMock()

        results = [
            SearchResult(
                identifier="img1",
                title="Test",
                thumbnail_url="https://archive.org/services/img/img1",
                details_url="",
            )
        ]

        with patch("llomax.search.thumbnails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await download_thumbnails(results)

        assert results[0].image is not None

    async def test_failed_download_leaves_image_none(self):
        results = [
            SearchResult(
                identifier="bad",
                title="Bad",
                thumbnail_url="https://archive.org/services/img/bad",
                details_url="",
            )
        ]

        with patch("llomax.search.thumbnails.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.HTTPError("fail"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await download_thumbnails(results)

        assert results[0].image is None

    async def test_skips_results_without_url(self):
        results = [
            SearchResult(identifier="no_url", title="No URL", thumbnail_url="", details_url="")
        ]
        await download_thumbnails(results)
        assert results[0].image is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use_response(tool_calls: list[dict], stop_reason="tool_use"):
    """Build a mock Anthropic response with tool_use content blocks."""
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


def _make_mcp_tool_result(text: str):
    result = MagicMock()
    result.content = [TextContent(type="text", text=text)]
    return result


@dataclass
class FakeMcpTool:
    name: str
    description: str
    inputSchema: dict


_FAKE_MCP_TOOLS = [
    FakeMcpTool(
        name="search_images_tool",
        description="Search images",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
]


# ---------------------------------------------------------------------------
# forward_tool_calls tests
# ---------------------------------------------------------------------------


class TestForwardToolCalls:
    async def test_forwards_tool_use_blocks(self):
        search_json = json.dumps(
            [
                {"identifier": "a1", "title": "A", "thumbnail_url": "", "details_url": ""},
            ]
        )
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=_make_mcp_tool_result(search_json))

        block = MagicMock()
        block.type = "tool_use"
        block.id = "t1"
        block.name = "search_images_tool"
        block.input = {"query": "test"}

        results_by_id: dict[str, SearchResult] = {}
        tool_results = await forward_tool_calls(mock_session, [block], results_by_id)

        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "t1"
        assert "a1" in results_by_id

    async def test_skips_non_tool_use_blocks(self):
        text_block = MagicMock()
        text_block.type = "text"

        mock_session = AsyncMock()
        results_by_id: dict[str, SearchResult] = {}
        tool_results = await forward_tool_calls(mock_session, [text_block], results_by_id)

        assert tool_results == []
        mock_session.call_tool.assert_not_called()

    async def test_non_search_tool_does_not_collect_results(self):
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_mcp_tool_result('[{"identifier":"col1","title":"C"}]')
        )

        block = MagicMock()
        block.type = "tool_use"
        block.id = "t1"
        block.name = "list_curated_collections_tool"
        block.input = {}

        results_by_id: dict[str, SearchResult] = {}
        await forward_tool_calls(mock_session, [block], results_by_id)

        assert results_by_id == {}


# ---------------------------------------------------------------------------
# run_agent_loop tests
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    async def test_returns_results_on_end_turn(self):
        search_json = json.dumps(
            [
                {"identifier": "r1", "title": "R", "thumbnail_url": "", "details_url": ""},
            ]
        )
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=_make_mcp_tool_result(search_json))

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[
                _make_tool_use_response(
                    [{"id": "t1", "name": "search_images_tool", "input": {"query": "q"}}]
                ),
                _make_end_turn_response(),
            ]
        )

        tools = mcp_tools_to_anthropic(_FAKE_MCP_TOOLS)
        results = await run_agent_loop(mock_client, "model", tools, mock_session, "prompt")

        assert "r1" in results
        assert mock_client.messages.create.call_count == 2

    async def test_caps_at_max_turns(self):
        search_json = json.dumps(
            [
                {"identifier": "x", "title": "X", "thumbnail_url": "", "details_url": ""},
            ]
        )
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=_make_mcp_tool_result(search_json))

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                [{"id": "t1", "name": "search_images_tool", "input": {"query": "loop"}}]
            )
        )

        tools = mcp_tools_to_anthropic(_FAKE_MCP_TOOLS)
        results = await run_agent_loop(mock_client, "model", tools, mock_session, "prompt")

        assert mock_client.messages.create.call_count == MAX_AGENT_TURNS
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# SearchAgent integration tests (fully mocked)
# ---------------------------------------------------------------------------


class TestSearchAgent:
    async def test_single_turn(self):
        """One tool call then end_turn — returns parsed results."""
        search_json = json.dumps(
            [
                {
                    "identifier": "a1",
                    "title": "Alpha",
                    "thumbnail_url": "http://x/a1",
                    "details_url": "",
                },
            ]
        )

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = _FAKE_MCP_TOOLS
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.call_tool = AsyncMock(return_value=_make_mcp_tool_result(search_json))

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _make_tool_use_response(
                    [{"id": "t1", "name": "search_images_tool", "input": {"query": "alpha"}}]
                ),
                _make_end_turn_response(),
            ]
        )

        agent = SearchAgent(anthropic_client=mock_anthropic)

        with (
            patch("llomax.search.mcp.stdio_client") as mock_stdio,
            patch("llomax.search.mcp.ClientSession") as mock_cs_cls,
            patch("llomax.search.thumbnails.download_thumbnails", new_callable=AsyncMock),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_cs_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            results = await agent.search("find alpha images")

        assert len(results) == 1
        assert results[0].identifier == "a1"

    async def test_deduplication(self):
        """Overlapping identifiers across searches are deduplicated."""
        search_json_1 = json.dumps(
            [
                {"identifier": "dup", "title": "First", "thumbnail_url": "", "details_url": ""},
                {"identifier": "unique1", "title": "U1", "thumbnail_url": "", "details_url": ""},
            ]
        )
        search_json_2 = json.dumps(
            [
                {"identifier": "dup", "title": "Second", "thumbnail_url": "", "details_url": ""},
                {"identifier": "unique2", "title": "U2", "thumbnail_url": "", "details_url": ""},
            ]
        )

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = _FAKE_MCP_TOOLS
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.call_tool = AsyncMock(
            side_effect=[
                _make_mcp_tool_result(search_json_1),
                _make_mcp_tool_result(search_json_2),
            ]
        )

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(
            side_effect=[
                _make_tool_use_response(
                    [{"id": "t1", "name": "search_images_tool", "input": {"query": "q1"}}]
                ),
                _make_tool_use_response(
                    [{"id": "t2", "name": "search_images_tool", "input": {"query": "q2"}}]
                ),
                _make_end_turn_response(),
            ]
        )

        agent = SearchAgent(anthropic_client=mock_anthropic)

        with (
            patch("llomax.search.mcp.stdio_client") as mock_stdio,
            patch("llomax.search.mcp.ClientSession") as mock_cs_cls,
            patch("llomax.search.thumbnails.download_thumbnails", new_callable=AsyncMock),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_cs_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            results = await agent.search("test")

        identifiers = [r.identifier for r in results]
        assert len(identifiers) == 3
        assert identifiers.count("dup") == 1
        dup_result = next(r for r in results if r.identifier == "dup")
        assert dup_result.title == "First"

    async def test_max_turns_safety(self):
        """Agent loop terminates after MAX_AGENT_TURNS even if LLM keeps requesting tools."""
        search_json = json.dumps(
            [
                {"identifier": "x", "title": "X", "thumbnail_url": "", "details_url": ""},
            ]
        )

        mock_session = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = _FAKE_MCP_TOOLS
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        mock_session.call_tool = AsyncMock(return_value=_make_mcp_tool_result(search_json))

        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                [{"id": "t1", "name": "search_images_tool", "input": {"query": "loop"}}]
            )
        )

        agent = SearchAgent(anthropic_client=mock_anthropic)

        with (
            patch("llomax.search.mcp.stdio_client") as mock_stdio,
            patch("llomax.search.mcp.ClientSession") as mock_cs_cls,
            patch("llomax.search.thumbnails.download_thumbnails", new_callable=AsyncMock),
        ):
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_cs_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            results = await agent.search("infinite loop")

        assert mock_anthropic.messages.create.call_count == MAX_AGENT_TURNS
        assert len(results) >= 1
