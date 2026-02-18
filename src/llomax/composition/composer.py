from __future__ import annotations

import random

from PIL import Image

from llomax.models import CollageOutput, EntityItem


def compose(
    elements: list[EntityItem],
    canvas_size: tuple[int, int] = (1024, 1024),
    background: Image.Image | None = None,
) -> CollageOutput:
    """Place each entity crop at a random position on a canvas.

    The canvas is initialised from ``background`` if provided, otherwise
    a solid white canvas is used. Elements larger than the canvas are
    pinned to ``(0, 0)``.

    Args:
        elements: Entity crops to place on the canvas.
        canvas_size: ``(width, height)`` in pixels.
        background: Optional background image. Resized to ``canvas_size``
            if its dimensions differ.

    Returns:
        A ``CollageOutput`` with the composed image, dimensions, and
        entity provenance records for every placed crop.
    """
    width, height = canvas_size

    if background is not None:
        canvas = background.resize((width, height)).convert("RGB")
    else:
        canvas = Image.new("RGB", (width, height), "white")

    provenance: list[dict] = []

    for element in elements:
        img = element.load_image()
        if img is None:
            continue

        max_x = max(0, width - img.width)
        max_y = max(0, height - img.height)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        canvas.paste(img, (x, y))

        provenance.append(
            {
                "item_id": element.item_id,
                "parent_image_id": element.parent_image_id,
                "label": element.label,
                "size": list(element.size),
                "position": [x, y],
                "metadata": element.metadata,
            }
        )

    return CollageOutput(image=canvas, width=width, height=height, entity_provenance=provenance)
