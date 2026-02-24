from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PIL import Image

from llomax.models import Fragment, SourceImage

if TYPE_CHECKING:
    import torch

_DEFAULT_MODEL_TYPE = "vit_b"
_DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"


class Segmenter:
    """SAM-based image segmenter with OpenVINO hardware acceleration.

    Uses the Segment Anything Model (SAM) ``AutomaticMaskGenerator`` to
    detect and extract all distinct visual elements from a ``SourceImage``.
    Each mask becomes a ``Fragment`` with a transparent (RGBA) background.

    Inference runs through the OpenVINO runtime, targeting the Intel Arc
    GPU via ``device_name="AUTO"`` (or an explicit ``"GPU"``). The SAM
    image encoder is exported to OpenVINO IR format on first use and cached
    to ``openvino_cache_dir``. If OpenVINO is unavailable the segmenter
    falls back to PyTorch CPU.

    Args:
        checkpoint_path: Path to the SAM model weights (``.pth`` file).
        model_type: SAM variant — ``"vit_b"``, ``"vit_l"``, or ``"vit_h"``.
        device: OpenVINO device string, e.g. ``"AUTO"``, ``"GPU"``,
            ``"NPU"``, or ``"CPU"``.
        openvino_cache_dir: Directory for the exported OpenVINO IR files.
            Defaults to a ``ov_cache`` subdirectory next to the checkpoint.
        min_mask_area: Minimum pixel area for a mask to become a Fragment.
        min_stability_score: SAM stability score threshold (0–1).
    """

    def __init__(
        self,
        checkpoint_path: str | Path = _DEFAULT_CHECKPOINT,
        model_type: str = _DEFAULT_MODEL_TYPE,
        device: str = "AUTO",
        openvino_cache_dir: Path | str | None = None,
        min_mask_area: int = 500,
        min_stability_score: float = 0.85,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._model_type = model_type
        self._device = device
        self._openvino_cache_dir = (
            Path(openvino_cache_dir)
            if openvino_cache_dir is not None
            else self._checkpoint_path.parent / "ov_cache"
        )
        self._min_mask_area = min_mask_area
        self._min_stability_score = min_stability_score
        self._mask_generator = None
        self._ov_compiled = None
        self._ov_output_key = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, sources: list[SourceImage]) -> list[Fragment]:
        """Implement the ``AnalysisClient`` protocol by running segmentation in a thread.

        SAM inference is CPU/GPU-bound, so each source is processed via
        ``asyncio.to_thread`` to avoid blocking the event loop.

        Args:
            sources: Source images with valid ``local_path`` values.

        Returns:
            All fragments extracted across all source images.
        """
        fragments: list[Fragment] = []
        for source in sources:
            source_fragments = await asyncio.to_thread(self.segment, source)
            fragments.extend(source_fragments)
        return fragments

    def segment(self, source: SourceImage) -> list[Fragment]:
        """Extract all visual segments from a source image.

        Args:
            source: Source image with a valid ``local_path``.

        Returns:
            List of ``Fragment`` objects, one per detected segment.
            Empty list if the image cannot be loaded or no segments pass
            the area and stability thresholds.
        """
        image = source.load_image()
        if image is None:
            logger.warning("Cannot segment %s: local_path unavailable.", source.external_id)
            return []

        rgb_array = np.array(image.convert("RGB"))
        generator = self._get_mask_generator()
        masks = generator.generate(rgb_array)

        return [
            self._mask_to_fragment(mask, rgb_array, source.external_id)
            for mask in masks
            if mask["area"] >= self._min_mask_area
        ]

    # ------------------------------------------------------------------
    # Mask generator construction
    # ------------------------------------------------------------------

    def _get_mask_generator(self):
        """Return the cached mask generator, building it on first access.

        Returns:
            The ``SamAutomaticMaskGenerator`` instance, initialised once and reused.
        """
        if self._mask_generator is None:
            self._mask_generator = self._build_mask_generator()
        return self._mask_generator

    def _build_mask_generator(self):
        """Build SAM mask generator, preferring the OpenVINO backend."""
        try:
            return self._build_openvino_generator()
        except Exception as exc:
            logger.warning("OpenVINO backend unavailable (%s); falling back to PyTorch CPU.", exc)
            return self._build_pytorch_generator()

    def _build_pytorch_generator(self):
        """Build SAM mask generator using PyTorch (CPU)."""
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[self._model_type](checkpoint=str(self._checkpoint_path))
        sam.eval()
        return SamAutomaticMaskGenerator(
            sam,
            pred_iou_thresh=0.85,
            stability_score_thresh=self._min_stability_score,
            min_mask_region_area=self._min_mask_area,
        )

    def _build_openvino_generator(self):
        """Build SAM mask generator using OpenVINO for the image encoder.

        Exports the SAM image encoder to OpenVINO IR on first call, then
        compiles it for the target device and patches it into the SAM model
        so ``SamAutomaticMaskGenerator`` uses accelerated inference.

        Returns:
            A ``SamAutomaticMaskGenerator`` backed by the OpenVINO encoder.
        """
        import openvino as ov
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        ir_dir = self._openvino_cache_dir
        ir_xml = ir_dir / "sam_encoder.xml"
        onnx_path = ir_dir / "sam_encoder.onnx"

        if not ir_xml.exists():
            ir_dir.mkdir(parents=True, exist_ok=True)
            self._export_encoder_to_openvino(onnx_path, ir_xml)

        core = ov.Core()
        self._ov_compiled = core.compile_model(str(ir_xml), device_name=self._device)
        self._ov_output_key = self._ov_compiled.output(0)
        logger.info("SAM encoder compiled on OpenVINO device: %s", self._device)

        sam = sam_model_registry[self._model_type](checkpoint=str(self._checkpoint_path))
        sam.eval()

        sam.image_encoder.forward = self._ov_forward

        return SamAutomaticMaskGenerator(
            sam,
            pred_iou_thresh=0.85,
            stability_score_thresh=self._min_stability_score,
            min_mask_region_area=self._min_mask_area,
        )

    def _ov_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the OpenVINO-compiled encoder forward pass.

        Replaces ``sam.image_encoder.forward`` so ``SamAutomaticMaskGenerator``
        uses the compiled model for inference.

        Args:
            x: Image tensor from SAM's preprocessing pipeline.

        Returns:
            Image embedding tensor produced by the OpenVINO compiled encoder.
        """
        import torch

        result = self._ov_compiled({0: x.detach().cpu().numpy()})[self._ov_output_key]
        return torch.from_numpy(result)

    # ------------------------------------------------------------------
    # OpenVINO export
    # ------------------------------------------------------------------

    def _export_encoder_to_openvino(self, onnx_path: Path, ir_xml: Path) -> None:
        """Export the SAM image encoder to OpenVINO IR format.

        Traces the ViT encoder through ONNX and converts to IR. The
        exported files are written to ``onnx_path`` and ``ir_xml``.

        Args:
            onnx_path: Destination path for the intermediate ONNX file.
            ir_xml: Destination path for the OpenVINO IR XML file.
        """
        import openvino as ov
        import torch
        from segment_anything import sam_model_registry

        logger.info("Exporting SAM encoder to OpenVINO IR (one-time setup)…")

        sam = sam_model_registry[self._model_type](checkpoint=str(self._checkpoint_path))
        encoder = sam.image_encoder.eval()

        dummy = torch.zeros(1, 3, 1024, 1024)
        torch.onnx.export(
            encoder,
            dummy,
            str(onnx_path),
            opset_version=16,
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        )

        core = ov.Core()
        model = core.read_model(str(onnx_path))
        ov.save_model(model, str(ir_xml))
        logger.info("SAM encoder saved to %s", ir_xml)

    # ------------------------------------------------------------------
    # Mask → Fragment conversion
    # ------------------------------------------------------------------

    def _mask_to_fragment(
        self,
        mask_data: dict,
        rgb_array: np.ndarray,
        source_id: str,
    ) -> Fragment:
        """Convert a SAM mask dictionary to a ``Fragment`` with transparent background.

        Args:
            mask_data: SAM mask dict with keys ``segmentation`` (bool H×W
                array) and ``bbox`` (XYWH integers).
            rgb_array: Full source image as ``(H, W, 3)`` uint8 array.
            source_id: ``SourceImage.external_id`` of the parent image.

        Returns:
            ``Fragment`` with an RGBA image cropped to the mask bounding
            box. Pixels outside the mask have alpha 0.
        """
        seg: np.ndarray = mask_data["segmentation"]  # bool (H, W)
        x, y, w, h = mask_data["bbox"]  # XYWH from SAM
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        crop_rgb = rgb_array[y1:y2, x1:x2]
        crop_mask = seg[y1:y2, x1:x2]

        rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
        rgba[..., :3] = crop_rgb
        rgba[..., 3] = np.where(crop_mask, 255, 0).astype(np.uint8)

        return Fragment(
            source_id=source_id,
            image_rgba=Image.fromarray(rgba, mode="RGBA"),
            bounding_box=(x1, y1, x2, y2),
        )
