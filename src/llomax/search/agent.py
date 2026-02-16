"""LLM-driven search agent for the Internet Archive."""

from __future__ import annotations

from pathlib import Path

import anthropic
from anthropic.types import MessageParam, ToolParam
from mcp import ClientSession

from llomax.models import SearchResult
from llomax.search.mcp import (
    forward_tool_calls,
    mcp_tools_to_anthropic,
    open_mcp_session,
)
from llomax.search.thumbnails import download_thumbnails

MAX_AGENT_TURNS = 15
"""Hard cap on LLM conversation turns to prevent runaway loops."""

_DEFAULT_MCP_SERVER_PATH = str(Path.home() / "mcp-servers/internet-archive-mcp")
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


async def run_agent_loop(
    client: anthropic.AsyncAnthropic,
    model: str,
    tools: list[ToolParam],
    session: ClientSession,
    prompt: str,
) -> dict[str, SearchResult]:
    """Run the multi-turn LLM agent loop.

    The loop sends the prompt to Claude with the available MCP tools,
    forwards any ``tool_use`` responses to the MCP session, and feeds
    results back until Claude emits ``end_turn`` or ``MAX_AGENT_TURNS``
    is reached.

    Args:
        client: Anthropic async client instance.
        model: Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        tools: Tool definitions in Anthropic format.
        session: An active MCP ``ClientSession``.
        prompt: The user's creative search prompt.

    Returns:
        A dict mapping identifier -> ``SearchResult``, deduplicated
        across all search calls (first occurrence wins).
    """
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

        tool_results = await forward_tool_calls(session, response.content, results_by_id)

        messages.append({"role": "assistant", "content": list(response.content)})
        messages.append({"role": "user", "content": tool_results})

    return results_by_id


class SearchAgent:
    """LLM-driven search agent that queries the Internet Archive for images.

    Uses Claude (Anthropic SDK) as the brain and a local MCP server for
    Internet Archive search tools. The agent makes multiple varied
    searches, deduplicates results, and downloads thumbnails.

    Attributes:
        mcp_server_path: Filesystem path to the IA MCP server project.
        model: Anthropic model identifier to use for the agent.
        client: Anthropic async client instance.
    """

    def __init__(
        self,
        mcp_server_path: str | Path = _DEFAULT_MCP_SERVER_PATH,
        model: str = _DEFAULT_MODEL,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ):
        """Initialize the search agent.

        Args:
            mcp_server_path: Path to the ``internet-archive-mcp`` project
                directory. Defaults to the local development path.
            model: Anthropic model identifier. Defaults to Claude Sonnet.
            anthropic_client: Optional pre-configured client. If ``None``,
                a new ``AsyncAnthropic`` is created (reads
                ``ANTHROPIC_API_KEY`` from the environment).
        """
        self.mcp_server_path = str(mcp_server_path)
        self.model = model
        self.client = anthropic_client or anthropic.AsyncAnthropic()

    async def search(self, prompt: str) -> list[SearchResult]:
        """Search the Internet Archive based on a creative prompt.

        Starts the MCP server, runs the LLM agent loop to gather
        results, deduplicates by identifier, and downloads thumbnails.

        Args:
            prompt: Creative text prompt describing the desired images.

        Returns:
            A list of ``SearchResult`` objects with downloaded thumbnail
            images (where available).
        """
        async with open_mcp_session(self.mcp_server_path) as session:
            mcp_tools = await session.list_tools()
            tools = mcp_tools_to_anthropic(mcp_tools.tools)

            results_by_id = await run_agent_loop(self.client, self.model, tools, session, prompt)

        results = list(results_by_id.values())
        await download_thumbnails(results)
        return results
