from __future__ import annotations

import random

from PIL import Image

from llomax.models import AnalysisResult, CollageOutput


def compose(
    elements: list[AnalysisResult],
    canvas_size: tuple[int, int] = (1024, 1024),
) -> CollageOutput:
    """Place each element at a random position on a white canvas.

    Elements that are larger than the canvas are pinned to (0, 0).

    Args:
        elements: Visual elements to place on the canvas.
        canvas_size: ``(width, height)`` in pixels.

    Returns:
        A ``CollageOutput`` with the composed image and dimensions.
    """
    width, height = canvas_size
    canvas = Image.new("RGB", (width, height), "white")

    for element in elements:
        img = element.image
        max_x = max(0, width - img.width)
        max_y = max(0, height - img.height)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        canvas.paste(img, (x, y))

    return CollageOutput(image=canvas, width=width, height=height)
