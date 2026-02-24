from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import anthropic
from loguru import logger

from llomax.core.hooks import PipelineState

_BACKGROUND_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are selecting the background image for an artistic collage.

Given a list of source images (with titles, descriptions, and the pixel dimensions of \
their largest available fragment) and the user's creative prompt, select the ONE source \
that works best as a full-canvas background.

Prefer: large, atmospheric, scenic, landscape, or abstract images.
Avoid: portraits of isolated small objects, or sources whose largest fragment is very small.

Return ONLY the identifier of the chosen source — a plain string, nothing else.\
"""


def select_best_background(
    anthropic_client: anthropic.AsyncAnthropic,
    model: str = _BACKGROUND_MODEL,
) -> Callable[[PipelineState], Awaitable[None]]:
    """Return an ``after_curation`` hook that selects a canvas background source.

    The hook analyses source metadata and fragment dimensions to pick the source
    best suited as a full-canvas background. Sets ``state.background_source_id``
    on the pipeline state when a valid source is identified.

    Uses text metadata only — no Vision API calls.

    Args:
        anthropic_client: Async Anthropic client for the selection call.
        model: Claude model to use.

    Returns:
        Async hook callable that accepts a ``PipelineState``.
    """

    async def hook(state: PipelineState) -> None:
        if not state.sources:
            logger.debug("[select_best_background] No sources in state — skipping.")
            return

        # Compute the largest fragment per source by bounding-box area.
        largest_by_source: dict[str, tuple[int, int]] = {}
        for frag in state.fragments:
            x1, y1, x2, y2 = frag.bounding_box
            w, h = x2 - x1, y2 - y1
            existing = largest_by_source.get(frag.source_id)
            if existing is None or w * h > existing[0] * existing[1]:
                largest_by_source[frag.source_id] = (w, h)

        sources_info = [
            {
                "identifier": s.external_id,
                "title": s.title,
                "description": (s.description or "")[:300],
                "year": s.metadata.get("year", ""),
                "creator": s.metadata.get("creator", ""),
                "largest_fragment_px": largest_by_source.get(s.external_id, (0, 0)),
            }
            for s in state.sources
        ]

        user_message = (
            f"Creative prompt: {state.prompt}\n\n"
            f"Canvas size: {state.canvas_size[0]}×{state.canvas_size[1]} pixels.\n\n"
            f"Sources:\n{json.dumps(sources_info, indent=2)}"
        )

        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=128,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = (
            "".join(b.text for b in response.content if b.type == "text")
            .strip()
            .strip('"')
            .strip("'")
        )

        valid_ids = {s.external_id for s in state.sources}
        if raw in valid_ids:
            state.background_source_id = raw
            logger.debug("[select_best_background] Selected background: {!r}", raw)
        else:
            logger.warning(
                "[select_best_background] LLM returned unknown source id {!r} — no background set.",
                raw,
            )

    return hook
