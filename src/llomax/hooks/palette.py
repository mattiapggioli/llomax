from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from loguru import logger
from PIL import Image, ImageEnhance, ImageOps

from llomax.core.hooks import PipelineState

PaletteMode = Literal["pastel", "vivid", "vintage", "faded"]


def color_grade(
    mode: PaletteMode = "pastel",
) -> Callable[[PipelineState], Awaitable[None]]:
    """Return a ``pre_composition`` hook that applies a unified colour palette.

    Every fragment image and the background pass through the same PIL
    transformation, ensuring visual coherence across all collage elements.
    The alpha channel of RGBA fragment images is preserved unchanged.

    Modes:
        pastel: Desaturated, high-key colours blended toward white.
        vivid: Boosted saturation and contrast.
        vintage: Sepia-toned greyscale.
        faded: Reduced contrast with a light grey wash.

    Args:
        mode: Palette transformation to apply to all assets.

    Returns:
        Async hook callable that accepts a ``PipelineState``.
    """

    async def hook(state: PipelineState) -> None:
        if state.background_image is not None:
            state.background_image = _apply_palette(
                state.background_image.convert("RGB"), mode
            )
            logger.debug("[color_grade] Applied {!r} to background image.", mode)

        for frag in state.fragments:
            frag.image_rgba = _apply_palette(frag.image_rgba, mode)

        if state.fragments:
            logger.debug(
                "[color_grade] Applied {!r} to {} fragment(s).",
                mode,
                len(state.fragments),
            )

    return hook


def _apply_palette(img: Image.Image, mode: PaletteMode) -> Image.Image:
    """Apply a named palette transformation to a PIL image.

    RGBA images have their colour channels transformed while the alpha channel
    is preserved. All other modes are converted to RGB for processing and
    restored afterward.

    Args:
        img: Input PIL image (any mode).
        mode: Palette transformation to apply.

    Returns:
        Transformed PIL image in the same mode as the input.
    """
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        tr, tg, tb = _transform_rgb(rgb, mode).split()
        return Image.merge("RGBA", (tr, tg, tb, a))

    original_mode = img.mode
    rgb = img.convert("RGB")
    result = _transform_rgb(rgb, mode)
    return result if original_mode == "RGB" else result.convert(original_mode)


def _transform_rgb(img: Image.Image, mode: PaletteMode) -> Image.Image:
    """Apply a colour transformation to an RGB PIL image.

    Args:
        img: RGB mode PIL image.
        mode: Palette transformation to apply.

    Returns:
        Transformed RGB PIL image.
    """
    match mode:
        case "pastel":
            img = ImageEnhance.Color(img).enhance(0.5)
            white = Image.new("RGB", img.size, (255, 255, 255))
            return Image.blend(img, white, 0.3)
        case "vivid":
            img = ImageEnhance.Color(img).enhance(1.8)
            return ImageEnhance.Contrast(img).enhance(1.3)
        case "vintage":
            gray = ImageOps.grayscale(img)
            r = gray.point(lambda p: min(255, int(p * 1.08)))
            g = gray.point(lambda p: min(255, int(p * 0.85)))
            b = gray.point(lambda p: min(255, int(p * 0.66)))
            return Image.merge("RGB", (r, g, b))
        case "faded":
            img = ImageEnhance.Contrast(img).enhance(0.7)
            overlay = Image.new("RGB", img.size, (200, 200, 200))
            return Image.blend(img, overlay, 0.2)
        case _:
            return img
