from __future__ import annotations

import json

import anthropic
from anthropic.types import ToolParam
from loguru import logger

from llomax.search.clients.internet_archive_client import (
    CURATED_COLLECTIONS,
    ImageResult,
    InternetArchiveClient,
)

MAX_AGENT_TURNS = 10

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_CURATED_COLLECTIONS_BLOCK = "\n".join(
    f"  - {c['identifier']}: {c['title']} — {c['description']}"
    for c in CURATED_COLLECTIONS
)

_SYSTEM_PROMPT = f"""\
You are a creative search agent for the Internet Archive. Your goal is to build \
a high-quality, diverse candidate pool for an art curator.

You have two tools:
1. **find_collections** — discover relevant IA collections using a short list of \
broad keywords for a single theme.
2. **search_images** — search for images by providing a list of synonyms or related \
terms. The system joins your list with OR automatically, so each term independently \
retrieves results.

## Curated collections (consult before searching)
These high-quality collections are always available. Use collection=<identifier> \
in search_images whenever a theme maps to one of them:
{_CURATED_COLLECTIONS_BLOCK}

## Rules
1. CORRECT SPELLING: Fix all misspellings from the user prompt before using any \
keyword (e.g. "astronaunts" → "astronauts", "forrest" → "forest").
2. ATOMIC DISCOVERY: When calling find_collections, use 1–2 broad keywords that \
describe the same concept. Do not combine unrelated themes in a single call.
   CORRECT:   keywords=["ghost", "spirit"]
   INCORRECT: keywords=["ghost", "sailor", "forest"]
3. BROAD SEARCH: When calling search_images, provide a list of synonyms and related \
terms. The system applies OR logic — more terms means a larger candidate pool for the \
curator to filter.
   CORRECT:   keywords=["astronaut", "cosmonaut", "spaceman", "spacewalk"]
   INCORRECT: keywords=["astronaut"]
4. AVOID AESTHETIC TERMS: Do not include subjective words like "vintage", "retro", \
"beautiful", or "cool" when searching curated collections — these terms rarely appear \
in original archive metadata.
5. DIVERSIFIED EXPLORATION: Perform at least 3–5 searches using different keywords, \
synonyms, time periods (date_filter), or specific collections.
6. SEARCH SCALE: Set max_results to the target count per search_images call.
7. FINAL RESPONSE: Once you have explored several distinct thematic angles, \
provide a summary of the collections and search terms used.\
"""

_PLANNER_SYSTEM_PROMPT = f"""\
You are a Strategic Research Planner for the Internet Archive. Your task is to \
design an autonomous, multi-angle search strategy for an art curator building a \
collage.

You have two tools:
1. **find_collections** — discover IA collections by keyword. Results are \
returned immediately for your use in planning.
2. **search_images** — register a search intent. Parameters are recorded into \
the search plan; the actual execution happens after planning. Each call returns \
a confirmation, not actual results.

## Curated collections (consult before searching)
These high-quality collections are always available. Use collection=<identifier> \
in search_images whenever a theme maps to one of them instead of running a \
generic collection discovery:
{_CURATED_COLLECTIONS_BLOCK}

## Rules
1. CORRECT SPELLING: Fix all misspellings from the user prompt before using any \
keyword (e.g. "astronaunts" → "astronauts", "forrest" → "forest").
2. ATOMIC DISCOVERY: When calling find_collections, use 1–2 broad keywords that \
describe the same concept. The IA engine treats multiple unrelated terms as AND, \
returning near-zero results.
   CORRECT:   keywords=["ocean", "sea"]
   INCORRECT: keywords=["ocean", "sailor", "storm"]
3. MAP TO CURATED COLLECTIONS: Before calling find_collections for a theme, check \
whether a curated collection above already covers it. If so, skip discovery and use \
collection=<identifier> directly in search_images.
4. BROAD SEARCH: When calling search_images, provide a list of synonyms and related \
terms. The system applies OR logic — every term independently finds results, creating \
a large candidate pool for the curator to filter.
   CORRECT:   keywords=["flower", "bloom", "blossom", "petal", "flora"]
   INCORRECT: keywords=["flower"]
5. AVOID AESTHETIC TERMS: Do not include words like "vintage", "retro", "beautiful", \
or "cool" in keywords for curated collections — they rarely appear in original archive \
metadata.
6. CALCULATE YOUR TARGET: Plan for a total candidate pool of 4× max_items. \
Distribute max_results across searches to reach this total.
7. DIVERSIFY: Cover distinct thematic angles, time periods, collections, and \
keyword variations. Do not repeat the same angle twice.
8. COMPLETE: Once the planned pool reaches approximately 4× max_items, stop \
planning and respond with a brief summary of the strategy.\
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
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "1–2 broad keywords describing a single theme. "
                        "The system joins them with OR."
                    ),
                },
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "search_images",
        "description": (
            "Search for images on the Internet Archive. "
            "Provide a list of synonyms or related terms — the system joins them with OR "
            "to maximise recall. Mediatype:image is automatically enforced."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of synonyms and related terms. "
                        "E.g. [\"botanical\", \"flora\", \"illustration\", \"plant\"]. "
                        "The system joins them with OR."
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

        Returns:
            Deduplicated list of ``ImageResult`` items collected across all agent turns.
        """
        results_by_id: dict[str, ImageResult] = {}
        user_content = f"The user wants {max_items} images for the final collage.\n\n{prompt}"
        messages: list = [{"role": "user", "content": user_content}]

        for _ in range(MAX_AGENT_TURNS):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            )

            self._log_agent_reasoning(response)

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
        target_pool = max_items * 4
        user_content = (
            f"max_items={max_items} (target candidate pool: ~{target_pool} total results across all searches).\n\n"
            f"{prompt}"
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

            self._log_agent_reasoning(response)

            if response.stop_reason == "end_turn":
                break

            tool_results = self._process_planning_tool_calls(response, plan)
            messages.append({"role": "assistant", "content": list(response.content)})
            messages.append({"role": "user", "content": tool_results})

        return plan

    def _log_tool_call(self, tool_name: str, tool_input: dict, result_text: str) -> None:
        """Log the input and outcome of a single tool call at DEBUG level.

        Args:
            tool_name: Name of the tool that was invoked.
            tool_input: Input parameters supplied to the tool.
            result_text: JSON string returned by the tool.
        """
        if tool_name == "find_collections":
            logger.debug(
                "[find_collections] keywords={!r}", tool_input.get("keywords", [])
            )
            collections = json.loads(result_text)
            if collections:
                for col in collections:
                    logger.debug(
                        "  collection found: {} — {!r}",
                        col.get("identifier", "?"),
                        col.get("title", ""),
                    )
            else:
                logger.debug("  (no collections found)")

        elif tool_name == "search_images":
            logger.debug(
                "[search_images] keywords={!r}, collection={}, date_filter={}, max_results={}",
                tool_input.get("keywords", []),
                tool_input.get("collection"),
                tool_input.get("date_filter"),
                tool_input.get("max_results"),
            )
            try:
                results = json.loads(result_text)
                if isinstance(results, list):
                    logger.debug("  {} result(s) returned", len(results))
            except Exception:
                pass

    def _log_agent_reasoning(self, response) -> None:
        """Log any free-text reasoning blocks present in the agent response.

        The agent interleaves ``text`` blocks with ``tool_use`` blocks. Text
        blocks contain the model's step-by-step reasoning and are logged at
        DEBUG level so they appear in the pipeline log without cluttering
        normal INFO output.

        Args:
            response: Anthropic API response whose ``content`` list may
                contain ``text`` and/or ``tool_use`` blocks.
        """
        for block in response.content:
            if block.type == "text" and block.text.strip():
                logger.debug("[agent reasoning] {}", block.text.strip())

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
            self._log_tool_call(block.name, block.input, result_text)

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

    def _build_plan_item(self, tool_input: dict) -> dict:
        """Build a search plan item dict from a ``search_images`` tool call input.

        Args:
            tool_input: Tool input dict containing at minimum a ``keywords`` key
                and optionally ``collection``, ``date_filter``, and ``max_results``.

        Returns:
            Plan item dict with ``keywords`` and any provided optional fields.
        """
        item: dict = {"keywords": tool_input["keywords"]}
        if tool_input.get("collection"):
            item["collection"] = tool_input["collection"]
        if tool_input.get("date_filter"):
            item["date_filter"] = tool_input["date_filter"]
        if tool_input.get("max_results") is not None:
            item["max_results"] = tool_input["max_results"]
        return item

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
                plan.append(self._build_plan_item(block.input))
                result_text = json.dumps({"status": "Search parameters recorded in the plan"})
            else:
                result_text = self._dispatch_tool(block.name, block.input)

            self._log_tool_call(block.name, block.input, result_text)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )
        return tool_results
