"""Curator agent that selects the best assets from search candidates."""

from __future__ import annotations

import json

import anthropic

_CURATOR_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """\
You are an art curator selecting images for a collage. Given a creative prompt \
and a list of candidate images from the Internet Archive, select the best ones \
based on:

1. **Relevance** to the creative prompt
2. **Visual diversity** — prefer a mix of subjects, styles, and time periods
3. **Aesthetic quality** — prefer items with descriptive titles and rich metadata

Return ONLY a JSON array of selected identifier strings. No explanation, no \
markdown fences, just the raw JSON array. Example: ["id1", "id2", "id3"]\
"""


async def select_assets(
    prompt: str,
    candidates: list[dict],
    anthropic_client: anthropic.AsyncAnthropic,
    max_items: int = 20,
) -> list[str]:
    """Select the best candidates for the collage via a single LLM call.

    Args:
        prompt: The user's creative prompt.
        candidates: Sanitized candidate dicts with keys:
            identifier, title, description, year.
        anthropic_client: Anthropic async client instance.
        max_items: Maximum number of items to select.

    Returns:
        List of selected identifier strings.
    """
    user_message = (
        f"Creative prompt: {prompt}\n\n"
        f"Select up to {max_items} images from these candidates:\n\n"
        + json.dumps(candidates, indent=2)
    )

    response = await anthropic_client.messages.create(
        model=_CURATOR_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    selected = json.loads(text)
    if not isinstance(selected, list):
        return []
    return [s for s in selected if isinstance(s, str)]
