from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from llomax.core.hooks import HookManager, PipelineState
from llomax.hooks.background import select_best_background
from llomax.hooks.llm_composer import _parse_placements, llm_compose
from llomax.hooks.palette import _apply_palette, color_grade
from llomax.models import CollageOutput, Fragment, SourceImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    sources: list[SourceImage] | None = None,
    fragments: list[Fragment] | None = None,
    prompt: str = "test prompt",
    canvas_size: tuple[int, int] = (200, 200),
) -> PipelineState:
    return PipelineState(
        prompt=prompt,
        canvas_size=canvas_size,
        sources=sources or [],
        fragments=fragments or [],
    )


def _make_source(eid: str, title: str = "", description: str = "") -> SourceImage:
    return SourceImage(
        external_id=eid,
        title=title or eid,
        description=description,
        local_path=None,
        metadata={"year": "1950", "creator": ""},
    )


def _make_fragment(source_id: str, label: str = "object", w: int = 50, h: int = 50) -> Fragment:
    rgba = Image.new("RGBA", (w, h), (100, 150, 200, 180))
    return Fragment(
        source_id=source_id,
        image_rgba=rgba,
        bounding_box=(0, 0, w, h),
        label=label,
    )


def _mock_anthropic(text: str) -> AsyncMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# HookManager
# ---------------------------------------------------------------------------


class TestHookManager:
    async def test_register_and_run(self):
        manager = HookManager()
        calls: list[str] = []

        async def my_hook(state: PipelineState) -> None:
            calls.append(state.prompt)

        manager.register("after_curation", my_hook)
        await manager.run("after_curation", _make_state(prompt="hello"))
        assert calls == ["hello"]

    async def test_hooks_run_in_order(self):
        manager = HookManager()
        order: list[str] = []

        async def hook_a(state: PipelineState) -> None:
            order.append("a")

        async def hook_b(state: PipelineState) -> None:
            order.append("b")

        manager.register("after_curation", hook_a)
        manager.register("after_curation", hook_b)
        await manager.run("after_curation", _make_state())
        assert order == ["a", "b"]

    async def test_run_with_no_hooks_does_not_error(self):
        manager = HookManager()
        await manager.run("after_curation", _make_state())  # must not raise

    def test_register_override_returned_by_get_override(self):
        manager = HookManager()

        async def my_override(state: PipelineState) -> None:
            pass

        manager.register_override("composition_strategy", my_override)
        assert manager.get_override("composition_strategy") is my_override

    def test_get_override_returns_none_when_not_registered(self):
        assert HookManager().get_override("composition_strategy") is None

    def test_has_hooks_false_when_empty(self):
        assert not HookManager().has_hooks("after_curation")

    def test_has_hooks_true_after_register(self):
        manager = HookManager()
        manager.register("after_curation", AsyncMock())
        assert manager.has_hooks("after_curation")

    async def test_hook_mutates_state(self):
        manager = HookManager()

        async def setter(state: PipelineState) -> None:
            state.background_source_id = "chosen"

        manager.register("after_curation", setter)
        state = _make_state()
        await manager.run("after_curation", state)
        assert state.background_source_id == "chosen"


# ---------------------------------------------------------------------------
# select_best_background
# ---------------------------------------------------------------------------


class TestSelectBestBackground:
    async def test_sets_background_source_id(self):
        src = _make_source("src1", title="Landscape Photo")
        state = _make_state(
            sources=[src],
            fragments=[_make_fragment("src1", w=300, h=200)],
        )
        hook = select_best_background(_mock_anthropic("src1"))
        await hook(state)
        assert state.background_source_id == "src1"

    async def test_ignores_unknown_id(self):
        src = _make_source("src1")
        state = _make_state(sources=[src], fragments=[_make_fragment("src1")])
        hook = select_best_background(_mock_anthropic("totally_unknown"))
        await hook(state)
        assert state.background_source_id is None

    async def test_empty_sources_no_crash(self):
        state = _make_state(sources=[], fragments=[])
        hook = select_best_background(_mock_anthropic(""))
        await hook(state)  # must not raise
        assert state.background_source_id is None

    async def test_strips_quotes_from_llm_response(self):
        src = _make_source("src2")
        state = _make_state(sources=[src], fragments=[_make_fragment("src2")])
        hook = select_best_background(_mock_anthropic('"src2"'))
        await hook(state)
        assert state.background_source_id == "src2"


# ---------------------------------------------------------------------------
# color_grade
# ---------------------------------------------------------------------------


class TestColorGrade:
    async def test_transforms_fragment_rgba(self):
        frag = _make_fragment("src1")
        original = list(frag.image_rgba.get_flattened_data())
        state = _make_state(fragments=[frag])
        await color_grade("pastel")(state)
        assert list(state.fragments[0].image_rgba.get_flattened_data()) != original

    async def test_preserves_alpha_channel(self):
        frag = _make_fragment("src1")
        original_alpha = [p[3] for p in frag.image_rgba.get_flattened_data()]
        state = _make_state(fragments=[frag])
        await color_grade("vintage")(state)
        new_alpha = [p[3] for p in state.fragments[0].image_rgba.get_flattened_data()]
        assert new_alpha == original_alpha

    async def test_transforms_background_image(self):
        bg = Image.new("RGB", (100, 100), (80, 120, 160))
        original = list(bg.get_flattened_data())
        state = _make_state()
        state.background_image = bg
        await color_grade("vivid")(state)
        assert list(state.background_image.get_flattened_data()) != original

    async def test_all_modes_run_without_error(self):
        for mode in ("pastel", "vivid", "vintage", "faded"):
            frag = _make_fragment("src1")
            state = _make_state(fragments=[frag])
            state.background_image = Image.new("RGB", (50, 50), (100, 100, 100))
            await color_grade(mode)(state)  # type: ignore[arg-type]

    async def test_no_crash_without_background(self):
        frag = _make_fragment("src1")
        state = _make_state(fragments=[frag])
        assert state.background_image is None
        await color_grade("faded")(state)  # must not raise

    def test_apply_palette_preserves_alpha_directly(self):
        img = Image.new("RGBA", (10, 10), (100, 150, 200, 128))
        result = _apply_palette(img, "pastel")
        assert result.mode == "RGBA"
        original_alpha = [p[3] for p in img.get_flattened_data()]
        result_alpha = [p[3] for p in result.get_flattened_data()]
        assert result_alpha == original_alpha


# ---------------------------------------------------------------------------
# llm_compose
# ---------------------------------------------------------------------------


class TestLlmCompose:
    def _placement_json(
        self, frag: Fragment, x: int = 10, y: int = 20, scale: float = 1.0
    ) -> str:
        return json.dumps(
            {frag.fragment_id: {"x": x, "y": y, "scale": scale, "reason": "artistic choice"}}
        )

    async def test_returns_collage_output(self):
        frag = _make_fragment("src1")
        state = _make_state(sources=[_make_source("src1")], fragments=[frag])
        hook = llm_compose(_mock_anthropic(self._placement_json(frag)))
        result = await hook(state)
        assert isinstance(result, CollageOutput)
        assert result.width == 200
        assert result.height == 200

    async def test_uses_llm_placement(self):
        frag = _make_fragment("src1", w=20, h=20)
        state = _make_state(sources=[_make_source("src1")], fragments=[frag])
        hook = llm_compose(_mock_anthropic(self._placement_json(frag, x=50, y=60)))
        result = await hook(state)
        assert result.fragment_provenance[0]["position"] == [50, 60]

    async def test_fallback_to_random_on_invalid_json(self):
        frag = _make_fragment("src1")
        state = _make_state(sources=[_make_source("src1")], fragments=[frag])
        hook = llm_compose(_mock_anthropic("this is not json"))
        result = await hook(state)  # must not raise
        assert isinstance(result, CollageOutput)
        assert len(result.fragment_provenance) == 1

    async def test_applies_scale(self):
        frag = _make_fragment("src1", w=40, h=40)
        state = _make_state(sources=[_make_source("src1")], fragments=[frag])
        payload = json.dumps(
            {frag.fragment_id: {"x": 0, "y": 0, "scale": 0.5, "reason": "small"}}
        )
        hook = llm_compose(_mock_anthropic(payload))
        result = await hook(state)
        assert result.fragment_provenance[0]["scale"] == 0.5

    async def test_llm_failure_falls_back_to_random(self):
        frag = _make_fragment("src1")
        state = _make_state(sources=[_make_source("src1")], fragments=[frag])
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=Exception("API error"))
        result = await llm_compose(client)(state)  # must not raise
        assert isinstance(result, CollageOutput)

    def test_parse_placements_strips_markdown_fences(self):
        frag_id = "abc-123"
        text = f'```json\n{{"{frag_id}": {{"x": 1, "y": 2, "scale": 1.0}}}}\n```'
        result = _parse_placements(text)
        assert frag_id in result
        assert result[frag_id]["x"] == 1

    def test_parse_placements_returns_empty_on_non_dict(self):
        assert _parse_placements('["not", "a", "dict"]') == {}

    def test_parse_placements_returns_empty_on_garbage(self):
        assert _parse_placements("not json at all !!!") == {}
