from __future__ import annotations

from llomax.models import Fragment, SourceImage


class PlaceholderAnnotator:
    """Simulates future Anthropic Vision annotation calls.

    Implements the "double analysis" structure: one pass for ``SourceImage``
    context (what the full archive item depicts) and a second pass for
    individual ``Fragment`` descriptions (what the extracted segment shows,
    informed by the parent item's metadata).

    Both methods return placeholder strings. Replace the bodies with live
    ``anthropic.AsyncAnthropic`` vision calls once a real backend is ready.
    """

    def annotate_source(self, source: SourceImage) -> str:
        """Build a placeholder context string for a ``SourceImage``.

        Simulates the first vision pass: a high-level description of the
        full archive item to give context for fragment-level analysis.

        Args:
            source: Source image with populated metadata.

        Returns:
            Placeholder context referencing the source title, year,
            creator, and description.
        """
        year = source.metadata.get("year", "unknown year")
        creator = source.metadata.get("creator", "unknown creator")
        snippet = source.description[:120] if source.description else "no description available"
        return (
            f"[Vision context placeholder] Source '{source.title}' "
            f"({year}) by {creator}. {snippet}"
        )

    def annotate_fragment(self, fragment: Fragment, source: SourceImage) -> str:
        """Build a placeholder description for a ``Fragment``.

        Simulates the second vision pass: a fine-grained description of
        the extracted region, referencing the parent source's context so
        that a future model can ground its answer.

        Args:
            fragment: The extracted fragment to describe.
            source: Parent ``SourceImage`` providing contextual metadata.

        Returns:
            Placeholder description combining fragment bounding box and
            parent source context.
        """
        x1, y1, x2, y2 = fragment.bounding_box
        w, h = x2 - x1, y2 - y1
        parent_context = source.description[:100] if source.description else "n/a"
        return (
            f"[Vision description placeholder] Fragment ({w}×{h}px) "
            f"at ({x1},{y1})–({x2},{y2}) from '{source.title}'. "
            f"Parent context: {parent_context}"
        )

    def annotate(self, sources: list[SourceImage], fragments: list[Fragment]) -> None:
        """Annotate all fragments in-place using their parent source context.

        For each fragment, looks up its parent ``SourceImage`` and sets
        ``fragment.description`` to the placeholder annotation string.
        ``fragment.label`` remains ``"unknown"`` until a real vision
        backend populates it.

        Args:
            sources: Source images used as a lookup table by ``external_id``.
            fragments: Fragments to annotate. Modified in place.
        """
        source_map = {s.external_id: s for s in sources}
        for fragment in fragments:
            source = source_map.get(fragment.source_id)
            if source is None:
                continue
            fragment.description = self.annotate_fragment(fragment, source)
