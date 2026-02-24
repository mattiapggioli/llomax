from __future__ import annotations

import json

import anthropic

from llomax.models import Fragment, SourceImage

_CURATOR_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """\
You are an art curator selecting source images for a collage. Each candidate has \
been pre-segmented — you can see how many visual elements (fragments) were detected \
and their types. Select sources based on:

1. **Relevance** — prefer sources whose detected fragment types match the creative prompt.
2. **Fragment quality** — prefer sources that yielded meaningful, clearly-labelled \
fragments. Sources with zero fragments contribute nothing to the collage and should \
generally be avoided.
3. **Visual diversity** — prefer a mix of fragment types, subjects, eras, and creators.
4. **Target count** — aim to select sources whose combined fragment count is close \
to the requested target.

Return ONLY a JSON array of selected external_id strings. No explanation, no \
markdown fences, just the raw JSON array. \
Example: ["img1", "img2"]\
"""


async def select_sources(
    prompt: str,
    candidates: list[SourceImage],
    fragments_by_source: dict[str, list[Fragment]],
    anthropic_client: anthropic.AsyncAnthropic,
    max_fragments: int = 20,
) -> list[str]:
    """Select the best source image candidates for the collage via a single LLM call.

    Each candidate summary includes pre-segmentation results so the model can
    select based on actual detected fragment types and counts rather than
    metadata alone.

    Args:
        prompt: The user's creative prompt.
        candidates: Available source images as ``SourceImage`` objects.
        fragments_by_source: Mapping of ``external_id`` to the fragments
            already extracted from each source image.
        anthropic_client: Anthropic async client instance.
        max_fragments: Target total number of fragments to collect across
            all selected sources.

    Returns:
        List of selected ``external_id`` strings.
    """
    summaries = [_source_summary(s, fragments_by_source.get(s.external_id, [])) for s in candidates]

    user_message = (
        f"Creative prompt: {prompt}\n\n"
        f"Target fragment count: ~{max_fragments}\n\n"
        f"Candidates (with pre-segmentation results):\n\n"
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


def _source_summary(source: SourceImage, fragments: list[Fragment]) -> dict:
    """Build the curator summary dict for a single source and its fragments.

    Args:
        source: The source image whose metadata to include.
        fragments: Fragments already extracted from this source.

    Returns:
        Dict with metadata fields plus ``fragment_count`` and
        ``fragment_labels`` (label → count mapping).
    """
    label_counts: dict[str, int] = {}
    for f in fragments:
        label_counts[f.label] = label_counts.get(f.label, 0) + 1
    return {
        "external_id": source.external_id,
        "title": source.title,
        "description": source.description[:200] if source.description else "",
        "year": source.metadata.get("year", ""),
        "creator": source.metadata.get("creator", ""),
        "fragment_count": len(fragments),
        "fragment_labels": label_counts,
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
