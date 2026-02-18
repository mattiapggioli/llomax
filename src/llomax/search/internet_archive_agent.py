from __future__ import annotations

import json

import anthropic
from anthropic.types import ToolParam

from llomax.search.clients.internet_archive_client import InternetArchiveClient, ImageResult

MAX_AGENT_TURNS = 10

_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """\
You are a creative search agent for the Internet Archive. Your goal is to build \
a high-quality, diverse candidate pool for an art curator.

You have two tools:
1. **find_collections** — discover relevant IA collections by keyword.
2. **search_images** — search for images using Lucene boolean syntax in the \
`keywords` field. You can optionally filter by collection, date range, and set \
max_results.

STRATEGY:
1. DIVERSIFIED EXPLORATION: Do not settle for one search. Even if your first \
query returns results, you MUST perform multiple additional searches (at least \
3-5) using different keywords, synonyms, time periods (date_filter), or specific \
collections (via find_collections).
2. SEARCH SCALE: For every 'search_images' call, set 'max_results' to the target \
count provided in the user's request. This ensures each specific angle you \
explore contributes a meaningful set of candidates.
3. QUALITY POOL: If the user wants 25 images, and you perform 4 diverse searches, \
you will provide a pool of ~100 candidates. This allows the Curator to select \
the absolute best items.
4. FINAL RESPONSE: Once you have explored several distinct thematic or visual \
angles, provide a summary of the collections and search terms used.\
"""

_PLANNER_SYSTEM_PROMPT = """\
You are a Research Planner for the Internet Archive. Your task is to design a \
multi-angle search strategy for an art curator building a collage.

You have two tools:
1. **find_collections** — discover relevant IA collections by keyword. Results \
are returned immediately for your use in planning.
2. **search_images** — register a search intent. Your call parameters are \
recorded into the search plan; the actual execution happens after planning. \
Each call returns a confirmation, not actual results.

STRATEGY:
1. DISCOVER: Use find_collections to identify relevant collections for the theme.
2. PLAN DIVERSIFIED SEARCHES: Register search_images calls (at minimum 3–5) \
using varied keywords, different collections, and different date ranges to cover \
distinct thematic angles.
3. SCALE: Set max_results on each search so the total planned pool is \
approximately 3–4× the requested max_items target.
4. COMPLETE: Once you have registered your searches, respond with a brief summary \
of the planned strategy.\
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
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results to return per call. "
                        "Set this to the target count from the user's request."
                    ),
                },
            },
            "required": ["keywords"],
        },
    },
]


class InternetArchiveAgent:
    """Agent that uses Claude with blinded IA tools."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        ia_client: InternetArchiveClient | None = None,
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
        self.ia_client = ia_client or InternetArchiveClient()

    async def search(self, prompt: str, max_items: int = 20) -> list[ImageResult]:
        """Run the agent loop and return deduplicated image results.

        Args:
            prompt: Creative text prompt describing the desired collage.
            max_items: Target number of images for the final collage.
        """
        results_by_id: dict[str, ImageResult] = {}
        user_content = (
            f"The user wants {max_items} images for the final collage.\n\n{prompt}"
        )
        messages: list = [{"role": "user", "content": user_content}]

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

            tool_results = self._process_tool_calls(response, results_by_id)
            messages.append({"role": "assistant", "content": list(response.content)})
            messages.append({"role": "user", "content": tool_results})

        return list(results_by_id.values())

    async def plan_search(self, prompt: str, max_items: int = 20) -> list[dict]:
        """Run the planner agent loop and return the accumulated search plan.

        The agent uses find_collections normally but search_images only registers
        search intents — no actual IA image searches are executed during planning.

        Args:
            prompt: Creative text prompt describing the desired collage.
            max_items: Target number of images for the final collage.

        Returns:
            List of search plan item dicts, each with at minimum a ``keywords``
            key and optionally ``collection``, ``date_filter``, and
            ``max_results``.
        """
        plan: list[dict] = []
        user_content = (
            f"The user wants {max_items} images for the final collage.\n\n{prompt}"
        )
        messages: list = [{"role": "user", "content": user_content}]

        for _ in range(MAX_AGENT_TURNS):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_PLANNER_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                break

            tool_results = self._process_planning_tool_calls(response, plan)
            messages.append({"role": "assistant", "content": list(response.content)})
            messages.append({"role": "user", "content": tool_results})

        return plan

    def _dispatch_tool(self, tool_name: str, tool_input: dict) -> str:
        """Route a tool call to the corresponding InternetArchiveClient method and return JSON.

        Args:
            tool_name: Name of the tool to dispatch.
            tool_input: Input parameters for the tool.

        Returns:
            JSON string with the tool result.
        """
        match tool_name:
            case "find_collections":
                results = self.ia_client.find_collections(keywords=tool_input["keywords"])
                return json.dumps(results)
            case "search_images":
                kwargs: dict = {
                    "keywords": tool_input["keywords"],
                    "collection": tool_input.get("collection"),
                    "date_filter": tool_input.get("date_filter"),
                }
                if tool_input.get("max_results") is not None:
                    kwargs["max_results"] = tool_input["max_results"]
                results = self.ia_client.search_images(**kwargs)
                return json.dumps(results)
            case _:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _collect_image_results(
        self, result_text: str, results_by_id: dict[str, ImageResult]
    ) -> None:
        """Parse search_images JSON and merge new results into the accumulator.

        Args:
            result_text: JSON string returned by a search_images tool call.
            results_by_id: Accumulator dict keyed by identifier. Modified in place.
        """
        for item in json.loads(result_text):
            ident = item.get("identifier", "")
            if not ident:
                continue
            results_by_id.setdefault(ident, item)

    def _process_tool_calls(self, response, results_by_id: dict[str, ImageResult]) -> list[dict]:
        """Execute tool calls from a response and return tool_result messages.

        Args:
            response: Anthropic API response containing tool_use blocks.
            results_by_id: Accumulator for image results. Modified in place.

        Returns:
            List of tool_result message dicts.
        """
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            result_text = self._dispatch_tool(block.name, block.input)

            if block.name == "search_images":
                self._collect_image_results(result_text, results_by_id)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )
        return tool_results

    def _process_planning_tool_calls(self, response, plan: list[dict]) -> list[dict]:
        """Process tool calls in planning mode.

        find_collections executes normally; search_images records parameters into
        the plan and returns a confirmation instead of actual results.

        Args:
            response: Anthropic API response containing tool_use blocks.
            plan: Accumulator for search plan items. Modified in place.

        Returns:
            List of tool_result message dicts.
        """
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "search_images":
                item: dict = {"keywords": block.input["keywords"]}
                if block.input.get("collection"):
                    item["collection"] = block.input["collection"]
                if block.input.get("date_filter"):
                    item["date_filter"] = block.input["date_filter"]
                if block.input.get("max_results") is not None:
                    item["max_results"] = block.input["max_results"]
                plan.append(item)
                result_text = json.dumps({"status": "Search parameters recorded in the plan"})
            else:
                result_text = self._dispatch_tool(block.name, block.input)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )
        return tool_results
