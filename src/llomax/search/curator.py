from __future__ import annotations

import json

import anthropic

from llomax.models import Fragment, SourceImage

_CURATOR_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """\
You are an art curator selecting individual visual fragments for a collage. Each \
fragment is a segment extracted from an Internet Archive source image — you may \
select any subset from any source, mixing fragments freely across sources.

Select fragments based on:

1. **Relevance** — prefer fragments whose label and source title match the creative prompt.
2. **Diversity** — draw from a variety of sources, labels, and visual types.
3. **Size** — prefer fragments with meaningful dimensions; very small ones rarely \
compose well.
4. **Target count** — select approximately the requested number of fragments in total.

Return ONLY a JSON array of selected fragment_id strings. No explanation, no \
markdown fences, just the raw JSON array. \
Example: ["id1", "id2"]\
"""


async def select_fragments(
    prompt: str,
    sources: list[SourceImage],
    fragments: list[Fragment],
    anthropic_client: anthropic.AsyncAnthropic,
    max_fragments: int = 20,
) -> list[str]:
    """Select individual fragments for the collage via a single LLM call.

    The curator receives a compact summary of every available fragment —
    including its detected label, pixel dimensions, and parent source context —
    and returns the ``fragment_id`` strings of the chosen subset. Fragments
    from different sources can be mixed freely.

    Args:
        prompt: The user's creative prompt.
        sources: All source images, used to attach title and metadata to
            each fragment summary.
        fragments: All extracted fragments across the full candidate pool.
        anthropic_client: Anthropic async client instance.
        max_fragments: Target number of fragments to select for composition.

    Returns:
        List of selected ``fragment_id`` strings.
    """
    source_map = {s.external_id: s for s in sources}
    summaries = [_fragment_summary(f, source_map.get(f.source_id)) for f in fragments]

    user_message = (
        f"Creative prompt: {prompt}\n\n"
        f"Target fragment count: ~{max_fragments}\n\n"
        f"Available fragments:\n\n"
        + json.dumps(summaries, indent=2)
    )

    response = await anthropic_client.messages.create(
        model=_CURATOR_MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    return _parse_identifiers(text)


def _fragment_summary(fragment: Fragment, source: SourceImage | None) -> dict:
    """Build a compact curator summary for a single fragment.

    Args:
        fragment: The fragment to summarise.
        source: Parent ``SourceImage`` for title and metadata context,
            or ``None`` if the source is no longer in the candidate list.

    Returns:
        Dict with ``fragment_id``, ``label``, pixel dimensions, and
        parent source fields (``source_id``, ``source_title``,
        ``source_year``, ``source_creator``).
    """
    x1, y1, x2, y2 = fragment.bounding_box
    return {
        "fragment_id": fragment.fragment_id,
        "label": fragment.label,
        "width": x2 - x1,
        "height": y2 - y1,
        "source_id": fragment.source_id,
        "source_title": source.title if source else "",
        "source_year": source.metadata.get("year", "") if source else "",
        "source_creator": source.metadata.get("creator", "") if source else "",
    }


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
