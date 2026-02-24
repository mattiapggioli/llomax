from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from PIL import Image

from llomax.models import Fragment, SourceImage


@dataclass
class PipelineState:
    """Shared mutable state passed between pipeline hook points.

    Attributes:
        prompt: The user's original creative prompt.
        canvas_size: ``(width, height)`` of the output canvas in pixels.
        sources: Source images selected by the curation stage.
        fragments: Fragments selected by the curation stage.
        background_source_id: ``external_id`` of the source flagged as the
            canvas background by an ``after_curation`` hook, or ``None``.
        background_image: PIL image loaded from the background source,
            populated by the pipeline after ``after_curation`` hooks run.
    """

    prompt: str
    canvas_size: tuple[int, int]
    sources: list[SourceImage]
    fragments: list[Fragment]
    background_source_id: str | None = None
    background_image: Image.Image | None = None


class HookManager:
    """Registry and executor for named pipeline hook points.

    Additive hooks (registered with ``register``) run in registration order
    and receive a ``PipelineState`` which they may mutate in place.

    Override hooks (registered with ``register_override``) replace the
    pipeline's default behaviour at that hook point entirely; the last
    registered override wins.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[[PipelineState], Awaitable[None]]]] = {}
        self._overrides: dict[str, Callable] = {}

    def register(
        self, hook_point: str, fn: Callable[[PipelineState], Awaitable[None]]
    ) -> None:
        """Register an additive hook for ``hook_point``.

        Args:
            hook_point: Name of the hook point (e.g. ``"after_curation"``).
            fn: Async callable that receives a ``PipelineState`` and mutates it.
        """
        self._hooks.setdefault(hook_point, []).append(fn)

    def register_override(self, hook_point: str, fn: Callable) -> None:
        """Register an override hook for ``hook_point``.

        The registered function replaces the pipeline's default implementation
        at that hook point. The last call to this method wins.

        Args:
            hook_point: Name of the hook point (e.g. ``"composition_strategy"``).
            fn: Async callable that receives a ``PipelineState`` and returns
                the hook point's result type.
        """
        self._overrides[hook_point] = fn

    async def run(self, hook_point: str, state: PipelineState) -> None:
        """Execute all additive hooks registered for ``hook_point`` in order.

        A no-op when no hooks are registered for the given point.

        Args:
            hook_point: Name of the hook point to execute.
            state: Pipeline state passed to each hook in turn.
        """
        for fn in self._hooks.get(hook_point, []):
            await fn(state)

    def get_override(self, hook_point: str) -> Callable | None:
        """Return the override callable for ``hook_point``, or ``None``.

        Args:
            hook_point: Name of the hook point.

        Returns:
            The registered override callable, or ``None`` if none is registered.
        """
        return self._overrides.get(hook_point)

    def has_hooks(self, hook_point: str) -> bool:
        """Return ``True`` if at least one additive hook is registered.

        Args:
            hook_point: Name of the hook point.

        Returns:
            ``True`` if at least one hook is registered for the point.
        """
        return bool(self._hooks.get(hook_point))
