from __future__ import annotations

import random

from PIL import Image

from llomax.models import CollageOutput, Fragment


def compose(
    fragments: list[Fragment],
    canvas_size: tuple[int, int] = (1024, 1024),
    background: Image.Image | None = None,
) -> CollageOutput:
    """Place each fragment at a random position on a canvas using alpha compositing.

    The canvas is initialised from ``background`` if provided, otherwise
    a solid white canvas is used. Fragments larger than the canvas are
    pinned to ``(0, 0)``. Each fragment's alpha channel is used as a mask
    so transparent regions of RGBA images blend correctly with the canvas.

    Args:
        fragments: Visual segments to place on the canvas.
        canvas_size: ``(width, height)`` in pixels.
        background: Optional background image. Resized to ``canvas_size``
            if its dimensions differ.

    Returns:
        A ``CollageOutput`` with the composed RGB image, dimensions, and
        provenance records for every placed fragment.
    """
    width, height = canvas_size

    if background is not None:
        canvas = background.resize((width, height)).convert("RGBA")
    else:
        canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    provenance: list[dict] = []

    for fragment in fragments:
        img = fragment.image_rgba
        max_x = max(0, width - img.width)
        max_y = max(0, height - img.height)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        canvas.paste(img, (x, y), mask=img.split()[3])

        provenance.append(
            {
                "source_id": fragment.source_id,
                "bounding_box": list(fragment.bounding_box),
                "label": fragment.label,
                "description": fragment.description,
                "position": [x, y],
            }
        )

    return CollageOutput(
        image=canvas.convert("RGB"),
        width=width,
        height=height,
        fragment_provenance=provenance,
    )
