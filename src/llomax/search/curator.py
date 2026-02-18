from __future__ import annotations

import json

import anthropic

from llomax.models import EntityItem

_CURATOR_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """\
You are an art curator selecting entity crops for a collage. Given a creative \
prompt and a list of candidate image crops detected from Internet Archive items, \
select the best ones based on:

1. **Relevance** to the creative prompt
2. **Visual diversity** — prefer a mix of entity labels, source images, and eras
3. **Aesthetic quality** — prefer crops from items with rich, descriptive titles

Return ONLY a JSON array of selected item_id strings. No explanation, no \
markdown fences, just the raw JSON array. \
Example: ["img1_person_0", "img2_chair_1"]\
"""


async def select_assets(
    prompt: str,
    candidates: list[EntityItem],
    anthropic_client: anthropic.AsyncAnthropic,
    max_items: int = 20,
) -> list[str]:
    """Select the best entity crop candidates for the collage via a single LLM call.

    Args:
        prompt: The user's creative prompt.
        candidates: Detected entity crops as ``EntityItem`` objects.
        anthropic_client: Anthropic async client instance.
        max_items: Maximum number of items to select.

    Returns:
        List of selected ``item_id`` strings.
    """
    summaries = [
        {
            "item_id": e.item_id,
            "label": e.label,
            "parent": e.parent_image_id,
            "title": e.metadata.get("title", ""),
            "year": e.metadata.get("year", ""),
            "size": list(e.size),
        }
        for e in candidates
    ]

    user_message = (
        f"Creative prompt: {prompt}\n\n"
        f"Select up to {max_items} entity crops from these candidates:\n\n"
        + json.dumps(summaries, indent=2)
    )

    response = await anthropic_client.messages.create(
        model=_CURATOR_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    return _parse_identifiers(text)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping a string.

    Args:
        text: Raw text that may be wrapped in triple-backtick fences.

    Returns:
        The inner content with fences removed and whitespace stripped.
    """
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_identifiers(text: str) -> list[str]:
    """Parse a JSON array of identifier strings from raw LLM output.

    Args:
        text: Raw LLM response text, possibly wrapped in markdown fences.

    Returns:
        List of valid string identifiers. Empty list on parse failure.
    """
    cleaned = _strip_markdown_fences(text)
    selected = json.loads(cleaned)
    if not isinstance(selected, list):
        return []
    return [s for s in selected if isinstance(s, str)]
