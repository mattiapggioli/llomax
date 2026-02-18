from __future__ import annotations

from pathlib import Path
from typing import Protocol

from llomax.models import EntityItem, SearchResult


class AnalysisClient(Protocol):
    """Protocol for image analysis backends.

    Implementations receive search results and return extracted visual
    elements as ``EntityItem`` objects with detection labels.
    """

    async def analyze(self, images: list[SearchResult]) -> list[EntityItem]:
        """Analyze images and extract labelled entity crops.

        Args:
            images: Search results with downloaded thumbnail images.

        Returns:
            A list of detected entity crops as ``EntityItem`` objects.
        """
        ...


class PlaceholderAnalysisClient:
    """Passthrough client that returns each image as a single "unknown" entity.

    Useful for testing the pipeline end-to-end before a real detection
    backend is available. No YOLO model is loaded; each full downloaded
    image is wrapped as an ``EntityItem`` with label ``"unknown"``.
    """

    async def analyze(self, images: list[SearchResult]) -> list[EntityItem]:
        """Return each image as an entity crop with label ``"unknown"``.

        Args:
            images: Search results to pass through. Results without an
                image are silently skipped.

        Returns:
            One ``EntityItem`` per input image that has a non-None image.
        """
        return [
            EntityItem(
                item_id=f"{img.identifier}_unknown_0",
                parent_image_id=img.identifier,
                size=(img.image.width, img.image.height),
                label="unknown",
                metadata={
                    "title": img.title,
                    "year": img.year,
                    "archive_url": img.details_url,
                    "creator": "",
                },
                file_path=None,
                image=img.image,
            )
            for img in images
            if img.image is not None
        ]


class YoloAnalysisClient:
    """YOLO-based object detection client with OpenVINO acceleration.

    Uses ``ultralytics`` to detect objects in downloaded images, crops
    each detection, saves the crops to disk, and returns them as
    ``EntityItem`` objects. The OpenVINO backend leverages Intel Arc
    GPU/NPU hardware when ``device="openvino"``.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "openvino",
        crops_dir: Path | str = Path("output/crops"),
        confidence: float = 0.25,
    ) -> None:
        """Initialize the YOLO analysis client.

        Args:
            model_name: YOLO model weights filename or path. The model
                is downloaded automatically on first use if not present.
            device: Inference device. Use ``"openvino"`` to export and
                run via OpenVINO (Intel Arc); use ``"cpu"`` for fallback.
            crops_dir: Directory where cropped entity images are saved.
            confidence: Minimum detection confidence threshold (0–1).
        """
        self.model_name = model_name
        self.device = device
        self.crops_dir = Path(crops_dir)
        self.confidence = confidence
        self._model = None

    def _get_model(self):
        """Load (and optionally export to OpenVINO) the YOLO model.

        Returns:
            A loaded ``YOLO`` model instance ready for inference.
        """
        if self._model is not None:
            return self._model

        from ultralytics import YOLO

        model = YOLO(self.model_name)

        if self.device == "openvino":
            ov_dir = Path(self.model_name).stem + "_openvino_model"
            if not Path(ov_dir).exists():
                model.export(format="openvino")
            model = YOLO(ov_dir)

        self._model = model
        return self._model

    async def analyze(self, images: list[SearchResult]) -> list[EntityItem]:
        """Detect objects in each image and return crops as EntityItems.

        For each detected object, the crop is saved to ``crops_dir`` and
        an ``EntityItem`` is created with the bounding box label and
        parent image provenance.

        Args:
            images: Search results with downloaded thumbnail images.
                Results without an image are silently skipped.

        Returns:
            A list of ``EntityItem`` objects, one per detected object
            across all input images.
        """
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        model = self._get_model()
        entity_items: list[EntityItem] = []

        for search_result in images:
            if search_result.image is None:
                continue

            results = model.predict(
                search_result.image,
                conf=self.confidence,
                verbose=False,
            )

            label_counts: dict[str, int] = {}
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    label = result.names[int(box.cls.item())]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    crop = search_result.image.crop((x1, y1, x2, y2))

                    idx = label_counts.get(label, 0)
                    label_counts[label] = idx + 1

                    item_id = f"{search_result.identifier}_{label}_{idx}"
                    file_path = self.crops_dir / f"{item_id}.png"
                    crop.save(file_path)

                    entity_items.append(
                        EntityItem(
                            item_id=item_id,
                            parent_image_id=search_result.identifier,
                            size=(crop.width, crop.height),
                            label=label,
                            metadata={
                                "title": search_result.title,
                                "year": search_result.year,
                                "archive_url": search_result.details_url,
                                "creator": "",
                            },
                            file_path=file_path,
                            image=crop,
                        )
                    )

        return entity_items
