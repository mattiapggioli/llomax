from __future__ import annotations

import json
import random
from collections.abc import Awaitable, Callable

import anthropic
from loguru import logger
from PIL import Image

from llomax.core.hooks import PipelineState
from llomax.models import CollageOutput

_COMPOSER_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are a Collage Artist placing visual fragments onto a canvas.

Given the creative prompt, the background description, and a list of fragments \
(with their labels, descriptions, and pixel dimensions), return a JSON object that \
maps each fragment_id to its placement on the canvas.

Format:
{
    "fragment_id_1": {"x": 100, "y": 200, "scale": 1.0, "reason": "..."},
    "fragment_id_2": {"x": 450, "y": 80,  "scale": 0.8, "reason": "..."}
}

Rules:
- x, y are the top-left pixel coordinates (integers, clamped to canvas bounds).
- scale multiplies the fragment's original size (float, 0.3 – 2.5).
- reason is a one-sentence artistic justification referencing the creative prompt.
- Spread fragments thoughtfully across the canvas — avoid piling everything centrally.
- Return ONLY the JSON object. No markdown fences, no extra text.\
"""


def agentic_composition(
    anthropic_client: anthropic.AsyncAnthropic,
    model: str = _COMPOSER_MODEL,
) -> Callable[[PipelineState], Awaitable[CollageOutput]]:
    """Return a ``composition_strategy`` override that uses an LLM to place fragments.

    The hook sends a text-based description of the canvas, the background, and
    every fragment to Claude and asks it to act as a Collage Artist. The LLM
    returns a JSON mapping each ``fragment_id`` to ``x``, ``y``, ``scale``, and a
    ``reason``. Fragments absent from the LLM response fall back to random
    placement. If the API call fails entirely, all fragments fall back to random.

    Uses text labels and metadata only — no Vision API calls.

    Args:
        anthropic_client: Async Anthropic client for the placement call.
        model: Claude model to use.

    Returns:
        Async hook callable that accepts a ``PipelineState`` and returns a
        ``CollageOutput``.
    """

    async def hook(state: PipelineState) -> CollageOutput:
        canvas_w, canvas_h = state.canvas_size

        if state.background_source_id:
            bg_src = next(
                (s for s in state.sources if s.external_id == state.background_source_id),
                None,
            )
            bg_desc = (
                f"'{bg_src.title}' — {(bg_src.description or '')[:200]}"
                if bg_src
                else "Background image (details unavailable)"
            )
        else:
            bg_desc = "None (white canvas)"

        fragment_descs = [
            {
                "fragment_id": frag.fragment_id,
                "label": frag.label,
                "description": (frag.description or "")[:200],
                "width_px": frag.image_rgba.width,
                "height_px": frag.image_rgba.height,
                "source_title": next(
                    (s.title for s in state.sources if s.external_id == frag.source_id), ""
                ),
            }
            for frag in state.fragments
        ]

        user_message = (
            f'Creative prompt: "{state.prompt}"\n\n'
            f"Canvas: {canvas_w}×{canvas_h} pixels.\n"
            f"Background: {bg_desc}\n\n"
            f"Fragments to place ({len(state.fragments)} total):\n"
            + json.dumps(fragment_descs, indent=2)
        )

        logger.debug("[agentic_composition] Requesting placements from LLM...")
        try:
            response = await anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = "".join(b.text for b in response.content if b.type == "text")
            placements = _parse_placements(raw)
            logger.debug(
                "[agentic_composition] {} placement(s) received from LLM.",
                len(placements),
            )
        except Exception as exc:
            logger.warning(
                "[agentic_composition] LLM call failed: {} — falling back to random placement.",
                exc,
            )
            placements = {}

        return _compose_with_placements(state, placements)

    return hook


def _parse_placements(text: str) -> dict[str, dict]:
    """Parse LLM placement JSON, stripping markdown fences if present.

    Args:
        text: Raw LLM response text.

    Returns:
        Dict mapping fragment_id to placement dict, or empty dict on failure.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        logger.warning("[agentic_composition] Could not parse placement JSON.")
        return {}


def _compose_with_placements(
    state: PipelineState,
    placements: dict[str, dict],
) -> CollageOutput:
    """Compose the collage using LLM-provided placements with random fallback.

    Fragments present in ``placements`` are placed at the specified coordinates
    with the given scale factor. Fragments not in ``placements`` are placed at
    random positions with scale 1.0.

    Args:
        state: Current pipeline state with fragments and background image.
        placements: Dict mapping fragment_id to placement dict from the LLM.

    Returns:
        Composed ``CollageOutput``.
    """
    canvas_w, canvas_h = state.canvas_size

    if state.background_image is not None:
        canvas = state.background_image.resize((canvas_w, canvas_h)).convert("RGBA")
    else:
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    provenance: list[dict] = []

    for frag in state.fragments:
        placement = placements.get(frag.fragment_id, {})
        scale = float(placement.get("scale", 1.0))
        scale = max(0.1, min(scale, 4.0))

        img = frag.image_rgba
        if scale != 1.0:
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        max_x = max(0, canvas_w - img.width)
        max_y = max(0, canvas_h - img.height)

        if "x" in placement and "y" in placement:
            x = max(0, min(int(placement["x"]), max_x))
            y = max(0, min(int(placement["y"]), max_y))
        else:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

        canvas.paste(img, (x, y), mask=img.split()[3])

        provenance.append(
            {
                "fragment_id": frag.fragment_id,
                "source_id": frag.source_id,
                "label": frag.label,
                "description": frag.description,
                "bounding_box": list(frag.bounding_box),
                "position": [x, y],
                "scale": scale,
                "reason": placement.get("reason", ""),
            }
        )

    return CollageOutput(
        image=canvas.convert("RGB"),
        width=canvas_w,
        height=canvas_h,
        fragment_provenance=provenance,
    )
