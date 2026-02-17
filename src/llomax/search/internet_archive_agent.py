from __future__ import annotations

import json

import anthropic
from anthropic.types import ToolParam

from llomax.search.clients.internet_archive_client import IAClient, ImageResult

MAX_AGENT_TURNS = 10

_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """\
You are a creative search agent for the Internet Archive. Your goal is to find \
diverse, high-quality images that match the user's creative prompt.

You have two tools:

1. **find_collections** — discover relevant IA collections by keyword.
2. **search_images** — search for images using Lucene boolean syntax in the \
`keywords` field (AND, OR, NOT, groupings with parentheses). You can optionally \
filter by collection and date range.

Strategy:
1. Optionally call find_collections to discover relevant collections.
2. Issue multiple search_images calls with varied keywords, synonyms, and \
   different angles to gather diverse material.
3. Use Lucene boolean syntax for precise queries: e.g. "(botanical OR flora) AND \
   illustration", "vintage AND (poster OR advertisement)".
4. When you have gathered enough results (aim for 10-20+ unique images), stop \
   and respond with a short text summary of what you found.\
"""

_TOOLS: list[ToolParam] = [
    {
        "name": "find_collections",
        "description": (
            "Search for Internet Archive collections by keyword. "
            "Returns collection identifiers, titles, and descriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Search keywords for finding collections.",
                },
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "search_images",
        "description": (
            "Search for images on the Internet Archive. Supports Lucene boolean "
            "syntax (AND, OR, NOT, groupings) in the keywords field. "
            "Mediatype:image is automatically enforced."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": (
                        "Search keywords using Lucene boolean syntax. "
                        "E.g. '(botanical OR flora) AND illustration'."
                    ),
                },
                "collection": {
                    "type": ["string", "null"],
                    "description": "Optional IA collection identifier to filter by.",
                },
                "date_filter": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional date range filter, e.g. '1900 TO 1950'. "
                        "Will be wrapped as date:[VALUE]."
                    ),
                },
            },
            "required": ["keywords"],
        },
    },
]


def _dispatch_tool(
    ia_client: IAClient,
    tool_name: str,
    tool_input: dict,
) -> str:
    """Route a tool call to the corresponding IAClient method and return JSON."""
    match tool_name:
        case "find_collections":
            results = ia_client.find_collections(keywords=tool_input["keywords"])
            return json.dumps(results)
        case "search_images":
            results = ia_client.search_images(
                keywords=tool_input["keywords"],
                collection=tool_input.get("collection"),
                date_filter=tool_input.get("date_filter"),
            )
            return json.dumps(results)
        case _:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})


class InternetArchiveAgent:
    """Agent that uses Claude with blinded IA tools."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        ia_client: IAClient | None = None,
    ):
        """Initialize the agent.

        Args:
            model: Claude model ID for the agent loop.
            anthropic_client: Anthropic async client. Created automatically
                if not provided.
            ia_client: Internet Archive client for executing tool calls.
                Created automatically if not provided.
        """
        self.model = model
        self.client = anthropic_client or anthropic.AsyncAnthropic()
        self.ia_client = ia_client or IAClient()

    async def search(self, prompt: str) -> list[ImageResult]:
        """Run the agent loop and return deduplicated image results."""
        results_by_id: dict[str, ImageResult] = {}
        messages: list = [{"role": "user", "content": prompt}]

        for _ in range(MAX_AGENT_TURNS):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                result_text = _dispatch_tool(self.ia_client, block.name, block.input)

                if block.name == "search_images":
                    for item in json.loads(result_text):
                        ident = item.get("identifier", "")
                        if ident:
                            results_by_id.setdefault(ident, item)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

            messages.append({"role": "assistant", "content": list(response.content)})
            messages.append({"role": "user", "content": tool_results})

        return list(results_by_id.values())
