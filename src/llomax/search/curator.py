from __future__ import annotations

import json

import anthropic

from llomax.models import SourceImage

_CURATOR_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """\
You are an art curator selecting source images for a collage. Given a creative \
prompt and a list of candidate images from the Internet Archive, select the best \
ones based on:

1. **Relevance** to the creative prompt
2. **Visual diversity** — prefer a mix of eras, creators, and subjects
3. **Aesthetic quality** — prefer images with rich, descriptive titles and metadata

Return ONLY a JSON array of selected external_id strings. No explanation, no \
markdown fences, just the raw JSON array. \
Example: ["img1", "img2"]\
"""


async def select_sources(
    prompt: str,
    candidates: list[SourceImage],
    anthropic_client: anthropic.AsyncAnthropic,
    max_sources: int = 10,
) -> list[str]:
    """Select the best source image candidates for the collage via a single LLM call.

    Args:
        prompt: The user's creative prompt.
        candidates: Available source images as ``SourceImage`` objects.
        anthropic_client: Anthropic async client instance.
        max_sources: Maximum number of source images to select.

    Returns:
        List of selected ``external_id`` strings.
    """
    summaries = [
        {
            "external_id": s.external_id,
            "title": s.title,
            "description": s.description[:200] if s.description else "",
            "year": s.metadata.get("year", ""),
            "creator": s.metadata.get("creator", ""),
        }
        for s in candidates
    ]

    user_message = (
        f"Creative prompt: {prompt}\n\n"
        f"Select up to {max_sources} source images from these candidates:\n\n"
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
