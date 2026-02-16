from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

import anthropic
from anthropic.types import MessageParam, ToolParam
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from PIL import Image

from llomax.models import SearchResult

logger = logging.getLogger(__name__)

MAX_AGENT_TURNS = 15

_DEFAULT_MCP_SERVER_PATH = "/home/mattiapggl/mcp-servers/internet-archive-mcp"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"

_SYSTEM_PROMPT = """\
You are a creative search agent for the Internet Archive. Your goal is to find \
diverse, high-quality images that match the user's creative prompt.

You have access to tools for searching the Internet Archive. Use them to:
1. Discover relevant collections with list_curated_collections_tool or search_collections_tool.
2. Issue multiple search_images_tool calls with varied queries and collections to gather diverse material.
3. Vary your search terms creatively — use synonyms, related concepts, and different angles.

When you are satisfied that you have gathered enough diverse results (aim for at \
least 10-20 unique images), stop and respond with a short text summary of what you found.\
"""


def _mcp_tools_to_anthropic(tools: list) -> list[ToolParam]:
    """Convert MCP Tool objects to Anthropic tool-use format."""
    return [
        ToolParam(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema,
        )
        for tool in tools
    ]


def _parse_search_results(raw: str) -> list[SearchResult]:
    """Parse a JSON string from search_images_tool into SearchResult objects."""
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


async def _download_thumbnails(results: list[SearchResult]) -> None:
    """Download thumbnail images for all results that have a thumbnail_url."""
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
                logger.warning(
                    "Failed to download thumbnail for %s", result.identifier
                )


@asynccontextmanager
async def _open_mcp_session(
    server_path: str,
) -> AsyncIterator[ClientSession]:
    """Start an MCP stdio server and yield an initialized ClientSession."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", server_path, "internet-archive-mcp"],
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def _forward_tool_calls(
    session: ClientSession,
    content_blocks: list,
    results_by_id: dict[str, SearchResult],
) -> list[dict]:
    """Forward tool_use blocks to MCP and collect search results.

    Returns the list of tool_result dicts to feed back to the LLM.
    """
    tool_results = []
    for block in content_blocks:
        if block.type != "tool_use":
            continue

        mcp_result = await session.call_tool(block.name, arguments=block.input)

        first = mcp_result.content[0] if mcp_result.content else None
        result_text = (
            first.text if isinstance(first, TextContent) else "[]"
        )

        if block.name == "search_images_tool":
            for sr in _parse_search_results(result_text):
                results_by_id.setdefault(sr.identifier, sr)

        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            }
        )
    return tool_results


async def _run_agent_loop(
    client: anthropic.AsyncAnthropic,
    model: str,
    tools: list[ToolParam],
    session: ClientSession,
    prompt: str,
) -> dict[str, SearchResult]:
    """Run the multi-turn LLM agent loop, returning deduplicated results."""
    results_by_id: dict[str, SearchResult] = {}
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for _ in range(MAX_AGENT_TURNS):
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            break

        tool_results = await _forward_tool_calls(
            session, response.content, results_by_id
        )

        messages.append({"role": "assistant", "content": list(response.content)})
        messages.append({"role": "user", "content": tool_results})

    return results_by_id


class SearchAgent:
    """LLM-driven search agent that queries the Internet Archive for images.

    Uses Claude (Anthropic SDK) as the brain and a local MCP server for
    Internet Archive search tools.
    """

    def __init__(
        self,
        mcp_server_path: str | Path = _DEFAULT_MCP_SERVER_PATH,
        model: str = _DEFAULT_MODEL,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ):
        self.mcp_server_path = str(mcp_server_path)
        self.model = model
        self.client = anthropic_client or anthropic.AsyncAnthropic()

    async def search(self, prompt: str) -> list[SearchResult]:
        """Run the agent loop: search IA via MCP, deduplicate, download thumbnails."""
        async with _open_mcp_session(self.mcp_server_path) as session:
            mcp_tools = await session.list_tools()
            tools = _mcp_tools_to_anthropic(mcp_tools.tools)

            results_by_id = await _run_agent_loop(
                self.client, self.model, tools, session, prompt
            )

        results = list(results_by_id.values())
        await _download_thumbnails(results)
        return results
