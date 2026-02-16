"""MCP client utilities for the Internet Archive stdio server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from anthropic.types import ToolParam, ToolResultBlockParam
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from llomax.models import SearchResult
from llomax.search.parsing import parse_search_results


@asynccontextmanager
async def open_mcp_session(server_path: str) -> AsyncIterator[ClientSession]:
    """Start the IA MCP server over stdio and yield an initialized session.

    The server subprocess is terminated when the context manager exits.

    Args:
        server_path: Filesystem path to the ``internet-archive-mcp``
            project directory (passed to ``uv run --directory``).

    Yields:
        An initialized ``ClientSession`` connected to the server.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", server_path, "internet-archive-mcp"],
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def mcp_tools_to_anthropic(tools: list) -> list[ToolParam]:
    """Convert MCP ``Tool`` objects to Anthropic tool-use format.

    The mapping is a direct 1:1 rename of ``inputSchema`` ->
    ``input_schema`` plus wrapping in ``ToolParam``.

    Args:
        tools: MCP Tool objects (from ``session.list_tools().tools``).

    Returns:
        A list of dicts conforming to the Anthropic ``tools`` parameter.
    """
    return [
        ToolParam(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema,
        )
        for tool in tools
    ]


async def forward_tool_calls(
    session: ClientSession,
    content_blocks: list,
    results_by_id: dict[str, SearchResult],
) -> list[ToolResultBlockParam]:
    """Forward LLM tool_use blocks to the MCP server.

    Each ``tool_use`` block is sent to ``session.call_tool``. Results
    from ``search_images_tool`` calls are parsed and deduplicated into
    *results_by_id* (first occurrence wins).

    Args:
        session: An active MCP ``ClientSession``.
        content_blocks: Anthropic response content blocks (may contain
            both ``text`` and ``tool_use`` types).
        results_by_id: Accumulator dict keyed by identifier. New search
            results are inserted via ``setdefault``. Modified in place.

    Returns:
        A list of ``tool_result`` dicts ready to append to the
        conversation messages.
    """
    tool_results: list[ToolResultBlockParam] = []
    for block in content_blocks:
        if block.type != "tool_use":
            continue

        mcp_result = await session.call_tool(block.name, arguments=block.input)

        first = mcp_result.content[0] if mcp_result.content else None
        result_text = first.text if isinstance(first, TextContent) else "[]"

        if block.name == "search_images_tool":
            for sr in parse_search_results(result_text):
                results_by_id.setdefault(sr.identifier, sr)

        tool_results.append(
            ToolResultBlockParam(
                type="tool_result",
                tool_use_id=block.id,
                content=result_text,
            )
        )
    return tool_results
