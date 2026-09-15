"""Object detection backends.

Two implementations satisfy :class:`ObjectDetector`:

* :class:`YoloDetector` wraps Ultralytics weights (YOLOv8/YOLOv11) for real
  deployments and for fine-tunes on dense retail datasets such as SKU110K.
* :class:`HeuristicShelfDetector` is a dependency-light OpenCV contour pipeline.
  It needs no weights and no GPU, produces the same value objects, and is the
  default so the repository is runnable and testable straight after clone.

Both are bound through :func:`build_detector`, so swapping backends is a config
change rather than a code change.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
import time
from typing import Any, Protocol, runtime_checkable
import urllib.request

import cv2
import numpy as np
from numpy.typing import NDArray

from smart_shelf.core.config import DetectorBackend, Settings
from smart_shelf.core.exceptions import DetectionError
from smart_shelf.core.logging import get_logger
from smart_shelf.cv.models import BoundingBox, Detection, DetectionResult, ShelfGeometry
from smart_shelf.cv.preprocessing import BGRImage, PreparedImage
from smart_shelf.cv.shelves import DEFAULT_ROW_TOLERANCE, cluster_rows, facing_indices

logger = get_logger(__name__)

DEFAULT_YOLO_WEIGHTS = "hf:chistopat/sku110k-yolo11-object-detector/weights/sku110k-yolo11-s640.pt"
"""SKU-110K-trained YOLO11s: single class "object", recall 0.867 / mAP50 0.927
on the 431k-instance SKU-110K test split. COCO checkpoints such as
``yolov8n.pt`` detect nothing on shelves, because none of their 80 classes is
"a packaged product"."""

HF_PREFIX = "hf:"
"""Weight spec scheme for a Hugging Face file: ``hf:<owner>/<repo>/<path>``."""

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"

MIN_FACING_PIXELS = 8
"""Narrower column spans than this are label strips or noise, never facings."""


@runtime_checkable
class ObjectDetector(Protocol):
    """Detects product-like objects in a prepared shelf image."""

    name: str

    def detect(self, image: PreparedImage) -> DetectionResult:  # pragma: no cover - protocol
        """Detect products in ``image``."""
        ...


def _non_max_suppression(
    candidates: list[tuple[BoundingBox, float]], iou_threshold: float
) -> list[tuple[BoundingBox, float]]:
    """Greedy NMS over ``(box, score)`` pairs, highest score first."""
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    kept: list[tuple[BoundingBox, float]] = []
    for box, score in ordered:
        if all(box.iou(other) < iou_threshold for other, _ in kept):
            kept.append((box, score))
    return kept


class HeuristicShelfDetector:
    """Contour + projection-profile product detector.

    The shelf rail runs behind every facing, so a plain contour pass welds a
    whole row into one blob. Detection therefore runs in two stages:

    1. Row runs: denoise, Canny, morphological close, then external contours.
       Each contour bounds either an isolated product or a whole merged row.
    2. Facing segmentation: inside a merged row, score each column by its
       vertical intensity spread. Columns crossing a product vary from backdrop
       to packaging; columns over an empty slot stay flat. Otsu splits the
       profile, contiguous occupied spans become facings, and each facing's
       vertical extent comes from the strongest horizontal edges in the span.

    Deterministic, and needs no weights or GPU. Use :class:`YoloDetector` with
    SKU110K-trained weights for production accuracy on dense shelves.
    """

    name = "opencv-heuristic"

    def __init__(
        self,
        *,
        min_area_ratio: float = 0.0008,
        max_area_ratio: float = 0.18,
        min_aspect: float = 0.15,
        max_aspect: float = 6.0,
        iou_threshold: float = 0.35,
        max_items: int = 200,
        confidence: float = 0.0,
        row_tolerance: float = DEFAULT_ROW_TOLERANCE,
        label: str = "product",
    ) -> None:
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        self._min_aspect = min_aspect
        self._max_aspect = max_aspect
        self._iou_threshold = iou_threshold
        self._max_items = max_items
        # The heuristic score is a rectangularity/occupancy heuristic, not a
        # calibrated probability, so it defaults to keeping every candidate.
        self._confidence = confidence
        self._row_tolerance = row_tolerance
        self._label = label

    # -- stage 1: merged row runs --------------------------------------------

    @staticmethod
    def _edge_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
        denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
        median = float(np.median(denoised))
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        edges = cv2.Canny(denoised, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return np.asarray(closed, dtype=np.uint8)

    # -- stage 2: facing segmentation inside a run ----------------------------

    @staticmethod
    def _contiguous_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
        """Return the ``[start, stop)`` spans of consecutive ``True`` values."""
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for index, flag in enumerate(mask):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                spans.append((start, index))
                start = None
        if start is not None:
            spans.append((start, len(mask)))
        return spans

    @staticmethod
    def _occupied_columns(roi: NDArray[np.float32]) -> tuple[NDArray[np.bool_], NDArray[np.uint8]]:
        """Split the run's columns into occupied and empty using Otsu.

        A column crossing a product swings from backdrop to packaging and back;
        a column over an empty slot stays flat. Rescaling that spread to 0-255
        and letting Otsu pick the split adapts to each row's own contrast
        instead of relying on a tuned constant.
        """
        spread = roi.std(axis=0)
        low, high = float(spread.min()), float(spread.max())
        if high <= low:
            # A perfectly uniform run: nothing here is a facing.
            flat = np.zeros(spread.shape, dtype=np.uint8)
            return np.zeros(spread.shape, dtype=np.bool_), flat

        scaled = ((spread - low) * (255.0 / (high - low))).astype(np.uint8)
        threshold, _ = cv2.threshold(
            scaled.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return scaled > threshold, scaled

    def _split_run(
        self, gray: NDArray[np.uint8], rect: tuple[int, int, int, int]
    ) -> list[tuple[BoundingBox, float]]:
        """Segment one merged row run into individual facings."""
        x, y, width, height = rect
        roi = gray[y : y + height, x : x + width].astype(np.float32)
        occupied, strength = self._occupied_columns(roi)
        min_span = max(MIN_FACING_PIXELS, int(height * 0.2))

        # On a densely packed row every column crosses some product, so the
        # occupancy split returns one span covering the whole run. The facing
        # pitch supplies the cut positions that contrast alone cannot.
        column_edges = np.abs(cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
        pitch = _dominant_pitch(
            _smooth(column_edges.astype(np.float32), width // 200),
            low=max(min_span, height * 0.15),
            high=width * 0.5,
        )

        boxes: list[tuple[BoundingBox, float]] = []
        for span_left, span_right in self._contiguous_runs(occupied):
            if span_right - span_left < min_span:
                continue
            for left, right in _subdivide(span_left, span_right, pitch):
                if right - left < min_span:
                    continue
                row_edges = np.abs(cv2.Sobel(roi[:, left:right], cv2.CV_32F, 0, 1, ksize=3)).sum(
                    axis=1
                )
                peak = float(row_edges.max())
                if peak <= 0:
                    continue
                rows = np.flatnonzero(row_edges > 0.3 * peak)
                if rows.size < 2:
                    continue
                score = round(min(0.95, 0.4 + 0.55 * float(strength[left:right].mean()) / 255.0), 4)
                boxes.append(
                    (
                        BoundingBox(
                            x1=float(x + left),
                            y1=float(y + int(rows[0])),
                            x2=float(x + right),
                            y2=float(y + int(rows[-1])),
                        ),
                        score,
                    )
                )
        return boxes

    def _row_bands(self, roi: NDArray[np.float32]) -> list[tuple[int, int]]:
        """Find product rows as the gaps between shelf rails.

        Rails, price strips and the shelf edge itself are long horizontal
        features, so they dominate row-wise horizontal-edge energy. What lies
        between two of them is a product row.
        """
        height, width = roi.shape[:2]
        energy = _smooth(
            np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1).astype(np.float32),
            height // 60,
        )
        rails = energy > energy.mean() + energy.std()

        bands: list[tuple[int, int]] = []
        cursor = 0
        for start, stop in self._contiguous_runs(rails):
            if start - cursor >= height * MIN_BAND_HEIGHT_RATIO:
                bands.append((cursor, start))
            cursor = stop
        if height - cursor >= height * MIN_BAND_HEIGHT_RATIO:
            bands.append((cursor, height))
        del width
        return bands

    def _dense_grid(
        self, gray: NDArray[np.uint8], rect: tuple[int, int, int, int]
    ) -> list[tuple[BoundingBox, float]]:
        """Segment a frame-filling run into a facing grid.

        Used when a contour swallows most of the image, which happens on real
        shelf photography where there is no visible backdrop to separate
        products from. Rows come from rail positions and columns from the
        facing pitch, so neither stage depends on background contrast.
        """
        x, y, width, height = rect
        roi = gray[y : y + height, x : x + width].astype(np.float32)
        min_span = max(MIN_FACING_PIXELS, int(width * 0.02))

        boxes: list[tuple[BoundingBox, float]] = []
        for top, bottom in self._row_bands(roi):
            band = roi[top:bottom]
            column_edges = _smooth(
                np.abs(cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0).astype(np.float32),
                width // 200,
            )
            pitch = _dominant_pitch(
                column_edges, low=max(min_span, (bottom - top) * 0.25), high=width * 0.5
            )
            if pitch is None:
                continue
            strength = float(column_edges.mean())
            ceiling = float(column_edges.max()) or 1.0
            for left, right in _subdivide(0, width, pitch):
                if right - left < min_span:
                    continue
                boxes.append(
                    (
                        BoundingBox(
                            x1=float(x + left),
                            y1=float(y + top),
                            x2=float(x + right),
                            y2=float(y + bottom),
                        ),
                        round(min(0.9, 0.35 + 0.5 * strength / ceiling), 4),
                    )
                )
        return boxes

    # -- assembly -------------------------------------------------------------

    def _accepts(self, box: BoundingBox, image_area: float) -> bool:
        if box.width <= 1 or box.height <= 1:
            return False
        area_ratio = box.area / image_area
        if not (self._min_area_ratio <= area_ratio <= self._max_area_ratio):
            return False
        aspect = box.width / box.height
        return self._min_aspect <= aspect <= self._max_aspect

    def _candidate_boxes(self, pixels: BGRImage) -> list[tuple[BoundingBox, float]]:
        height, width = pixels.shape[:2]
        image_area = float(height * width)
        gray = np.asarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        contours, _ = cv2.findContours(
            self._edge_mask(gray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[tuple[BoundingBox, float]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1 or (w * h) / image_area < self._min_area_ratio:
                continue

            box = BoundingBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))
            if self._accepts(box, image_area):
                # Rectangularity: packaged goods fill their bounding box; shelf
                # edges, price rails and label strips generally do not.
                fill = float(cv2.contourArea(contour)) / box.area
                candidates.append((box, round(min(0.99, 0.45 + 0.5 * fill), 4)))
                continue

            # A run that fills the frame has no usable backdrop, so the
            # occupancy split degenerates into noise; fall back to the grid.
            segmenter = (
                self._dense_grid if (w * h) / image_area >= DENSE_RUN_RATIO else self._split_run
            )
            candidates.extend(
                (split_box, score)
                for split_box, score in segmenter(gray, (x, y, w, h))
                if self._accepts(split_box, image_area)
            )
        return candidates

    def detect(self, image: PreparedImage) -> DetectionResult:
        """Run the two-stage pipeline over a prepared image."""
        started = time.perf_counter()
        try:
            candidates = self._candidate_boxes(image.pixels)
        except cv2.error as exc:  # pragma: no cover - guards malformed arrays
            raise DetectionError(
                "OpenCV detection pipeline failed", details={"cause": str(exc)}
            ) from exc

        above_threshold = [(box, score) for box, score in candidates if score >= self._confidence]
        kept = _non_max_suppression(above_threshold, self._iou_threshold)[: self._max_items]
        boxes = [box for box, _ in kept]
        levels, geometry = cluster_rows(
            boxes, image_height=image.height, tolerance=self._row_tolerance
        )
        slots = facing_indices(boxes, levels)

        detections = [
            Detection(
                box=box,
                label=self._label,
                confidence=score,
                shelf_level=level,
                facing_index=slot,
            )
            for (box, score), level, slot in zip(kept, levels, slots, strict=True)
        ]
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "detection.completed",
            backend=self.name,
            detections=len(detections),
            rows=geometry.row_count,
            duration_ms=round(duration_ms, 2),
        )
        return DetectionResult(
            detections=detections,
            image_width=image.width,
            image_height=image.height,
            backend=self.name,
            duration_ms=duration_ms,
            geometry=geometry,
        )


DENSE_SPLIT_FACTOR = 1.6
"""A span wider than this multiple of the facing pitch is treated as merged."""

MIN_PERIODICITY = 0.12
"""Autocorrelation floor below which a row is not treated as periodic."""

DENSE_RUN_RATIO = 0.55
"""A contour covering this much of the frame means the backdrop test is useless."""

MIN_BAND_HEIGHT_RATIO = 0.08
"""Shorter horizontal bands are price rails and label strips, not product rows."""


def _smooth(values: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    """Box-filter a 1-D profile, keeping its length."""
    width = max(3, int(window) | 1)
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _dominant_pitch(profile: NDArray[np.float32], low: float, high: float) -> int | None:
    """Estimate the repeat distance between facings, in pixels.

    Shelves are merchandised in near-uniform columns, so a row's vertical-edge
    profile is close to periodic. Autocorrelating it recovers that period even
    where individual facings cannot be segmented by contrast, which is what
    makes a densely packed row splittable at all.

    Returns:
        The pitch in pixels, or ``None`` when the profile shows no periodicity
        strong enough to act on.
    """
    centred = profile - profile.mean()
    if not np.any(centred):
        return None
    correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    peak = float(correlation[0])
    if peak <= 0:
        return None
    correlation = correlation / peak

    start, stop = int(low), int(min(high, len(correlation) - 1))
    if stop <= start:
        return None
    best = int(np.argmax(correlation[start:stop])) + start
    return best if float(correlation[best]) > MIN_PERIODICITY else None


def _subdivide(left: int, right: int, pitch: int | None) -> list[tuple[int, int]]:
    """Cut an over-wide span into equal facing-sized slices."""
    width = right - left
    if pitch is None or pitch <= 0 or width < pitch * DENSE_SPLIT_FACTOR:
        return [(left, right)]
    slices = max(2, round(width / pitch))
    edges = [left + round(width * index / slices) for index in range(slices + 1)]
    return list(pairwise(edges))


def _host_array(values: Any, dtype: type) -> NDArray[Any]:
    """Convert an Ultralytics box attribute to a numpy array.

    Those attributes are torch tensors. A tensor living on a CUDA device cannot
    be read by numpy until it is copied to host memory, and test doubles hand
    back plain arrays, so the copy is attempted rather than assumed.
    """
    to_cpu = getattr(values, "cpu", None)
    if callable(to_cpu):
        values = to_cpu()
    return np.asarray(values, dtype=dtype)


def _is_published_asset(weights: str) -> bool:
    """True when ``weights`` is a bare checkpoint name Ultralytics publishes."""
    return "/" not in weights and "\\" not in weights and weights.endswith(".pt")


def _parse_hf_spec(weights: str) -> tuple[str, str]:
    """Split ``hf:<owner>/<repo>/<path>`` into its repository and file path."""
    parts = weights[len(HF_PREFIX) :].split("/")
    if len(parts) < 3:
        raise DetectionError(
            "malformed Hugging Face weights spec",
            details={"weights": weights, "expected": "hf:<owner>/<repo>/<path-to.pt>"},
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def _download_from_hub(weights: str, target: Path) -> None:
    """Fetch a checkpoint from the Hugging Face CDN into ``target``.

    Uses a plain HTTPS request rather than ``huggingface_hub`` so that a public
    checkpoint costs no extra dependency.
    """
    repo, path = _parse_hf_spec(weights)
    url = HF_RESOLVE.format(repo=repo, path=path)
    # The URL is built from a fixed https template, never from caller input.
    request = urllib.request.Request(url, headers={"User-Agent": "smart-shelf-analytics"})  # noqa: S310
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        partial.write_bytes(response.read())
    partial.replace(target)


def ensure_weights(weights: str, weights_dir: Path) -> str:
    """Return a local path to ``weights``, downloading it if necessary.

    Resolution order: an existing path as given, then the cache directory, then
    a download from the Ultralytics release assets.

    Args:
        weights: a checkpoint name such as ``yolov8n.pt`` or a path to a file.
        weights_dir: directory used to cache downloaded checkpoints.

    Returns:
        A filesystem path Ultralytics can load.

    Raises:
        DetectionError: an explicit path does not exist, or the download failed.
    """
    from_hub = weights.startswith(HF_PREFIX)
    given = Path(weights).expanduser()
    if not from_hub and given.is_file():
        return str(given)

    name = _parse_hf_spec(weights)[1].rsplit("/", 1)[-1] if from_hub else given.name
    cached = weights_dir.expanduser() / name
    if cached.is_file():
        return str(cached)

    if not (from_hub or _is_published_asset(weights)):
        # An explicit path cannot be invented, and guessing an asset name from
        # it would silently load the wrong model.
        raise DetectionError(
            f"YOLO weights not found at {given}",
            details={"weights": weights, "cache_dir": str(weights_dir)},
        )

    logger.info("yolo.weights_download_started", weights=weights, target=str(cached))
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        if from_hub:
            _download_from_hub(weights, cached)
        else:
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(str(cached))
    except Exception as exc:
        raise DetectionError(
            f"could not download YOLO weights '{weights}'",
            details={
                "weights": weights,
                "cache_dir": str(weights_dir),
                "cause": str(exc),
                "hint": (
                    "First use needs network access. Pre-seed the cache "
                    "directory, point SHELF_YOLO_WEIGHTS at a local file, or "
                    "set SHELF_DETECTOR_BACKEND=heuristic."
                ),
            },
        ) from exc

    if not cached.is_file():
        raise DetectionError(
            f"download of '{weights}' reported success but produced no file",
            details={"expected": str(cached)},
        )
    logger.info(
        "yolo.weights_ready", weights=weights, path=str(cached), bytes=cached.stat().st_size
    )
    return str(cached)


class YoloDetector:
    """Ultralytics YOLO wrapper.

    Missing checkpoints are downloaded on first construction and cached, so a
    fresh install can select this backend with no extra setup.

    ``ultralytics`` is imported lazily despite being a required dependency: it
    pulls in torch, which costs seconds of import time that the OpenCV backend,
    the API and the CLI should not pay for.
    """

    name = "yolo"

    def __init__(
        self,
        weights: str = DEFAULT_YOLO_WEIGHTS,
        *,
        confidence: float = 0.25,
        max_items: int = 200,
        row_tolerance: float = DEFAULT_ROW_TOLERANCE,
        weights_dir: Path | None = None,
        model: Any | None = None,
    ) -> None:
        self._weights = weights
        self._confidence = confidence
        self._max_items = max_items
        self._row_tolerance = row_tolerance
        self.name = f"yolo:{weights}"
        if model is not None:
            self._model = model
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - ultralytics is required
            raise DetectionError(
                "ultralytics is not importable; reinstall the project dependencies "
                "or set SHELF_DETECTOR_BACKEND=heuristic",
                details={"weights": weights, "cause": str(exc)},
            ) from exc

        cache = weights_dir or (Path.home() / ".cache" / "smart-shelf-analytics" / "weights")
        resolved = ensure_weights(weights, cache)
        try:
            self._model = YOLO(resolved)
        except Exception as exc:
            raise DetectionError(
                f"could not load YOLO weights from {resolved}",
                details={"weights": weights, "cause": str(exc)},
            ) from exc

    def _names(self) -> dict[int, str]:
        names = getattr(self._model, "names", {}) or {}
        return {int(k): str(v) for k, v in dict(names).items()}

    def detect(self, image: PreparedImage) -> DetectionResult:
        """Run YOLO inference and map the predictions onto shelf rows."""
        started = time.perf_counter()
        try:
            predictions = self._model.predict(image.pixels, conf=self._confidence, verbose=False)
        except Exception as exc:
            raise DetectionError("YOLO inference failed", details={"cause": str(exc)}) from exc

        names = self._names()
        pairs: list[tuple[BoundingBox, float, str]] = []
        for prediction in predictions:
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            xyxy = _host_array(boxes.xyxy, float)
            confs = _host_array(boxes.conf, float).reshape(-1)
            classes = _host_array(boxes.cls, int).reshape(-1)
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes, strict=True):
                pairs.append(
                    (
                        BoundingBox(
                            x1=max(0.0, float(x1)),
                            y1=max(0.0, float(y1)),
                            x2=max(0.0, float(x2)),
                            y2=max(0.0, float(y2)),
                        ),
                        float(min(1.0, max(0.0, conf))),
                        names.get(int(cls), f"class_{int(cls)}"),
                    )
                )

        pairs.sort(key=lambda item: item[1], reverse=True)
        pairs = pairs[: self._max_items]
        boxes_only = [box for box, _, _ in pairs]
        levels, geometry = cluster_rows(
            boxes_only, image_height=image.height, tolerance=self._row_tolerance
        )
        slots = facing_indices(boxes_only, levels)

        detections = [
            Detection(box=box, label=label, confidence=conf, shelf_level=level, facing_index=slot)
            for (box, conf, label), level, slot in zip(pairs, levels, slots, strict=True)
        ]
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "detection.completed",
            backend=self.name,
            detections=len(detections),
            rows=geometry.row_count,
            duration_ms=round(duration_ms, 2),
        )
        return DetectionResult(
            detections=detections,
            image_width=image.width,
            image_height=image.height,
            backend=self.name,
            duration_ms=duration_ms,
            geometry=geometry or ShelfGeometry(row_count=0),
        )


class HybridDetector:
    """Runs YOLO, and falls back to the OpenCV pipeline when it comes up short.

    Two situations make the fallback worth having. The checkpoint may be
    unavailable, because the first run is still downloading it or the host is
    offline; and a checkpoint trained on one kind of shelf may return almost
    nothing on another. In both cases a weak result is better than no result,
    so whichever backend found more facings wins.
    """

    name = "hybrid"

    def __init__(
        self,
        primary: ObjectDetector,
        fallback: ObjectDetector,
        *,
        min_detections: int = 4,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._min_detections = min_detections
        self.name = f"hybrid:{getattr(primary, 'name', '?')}+{getattr(fallback, 'name', '?')}"

    def detect(self, image: PreparedImage) -> DetectionResult:
        """Detect with the primary backend, falling back when it underperforms."""
        try:
            primary = self._primary.detect(image)
        except DetectionError as exc:
            logger.warning("detector.primary_failed", error=exc.message, falling_back=True)
            return self._fallback.detect(image)

        if primary.count >= self._min_detections:
            return primary

        fallback = self._fallback.detect(image)
        logger.info(
            "detector.fallback_consulted",
            primary=primary.count,
            fallback=fallback.count,
            threshold=self._min_detections,
        )
        return fallback if fallback.count > primary.count else primary


def _build_heuristic(settings: Settings) -> HeuristicShelfDetector:
    return HeuristicShelfDetector(
        max_items=settings.detection_max_items,
        confidence=settings.detection_confidence,
        row_tolerance=settings.row_merge_tolerance,
    )


def _build_yolo(settings: Settings) -> YoloDetector:
    return YoloDetector(
        settings.yolo_weights,
        confidence=settings.detection_confidence,
        max_items=settings.detection_max_items,
        row_tolerance=settings.row_merge_tolerance,
        weights_dir=settings.yolo_weights_dir,
    )


def build_detector(settings: Settings) -> ObjectDetector:
    """Instantiate the detector named by ``settings.detector_backend``."""
    match settings.detector_backend:
        case DetectorBackend.YOLO:
            return _build_yolo(settings)
        case DetectorBackend.HYBRID:
            try:
                primary: ObjectDetector = _build_yolo(settings)
            except DetectionError as exc:
                # Weights unavailable: degrade to OpenCV rather than refusing.
                logger.warning("detector.yolo_unavailable", error=exc.message)
                return _build_heuristic(settings)
            return HybridDetector(
                primary,
                _build_heuristic(settings),
                min_detections=settings.hybrid_min_detections,
            )
        case _:
            return _build_heuristic(settings)
