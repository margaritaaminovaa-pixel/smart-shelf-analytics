"""Heuristic and YOLO detector behaviour."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from smart_shelf.core.config import DetectorBackend, Settings, VLMBackend
from smart_shelf.core.exceptions import DetectionError
from smart_shelf.cv.detector import (
    DEFAULT_YOLO_WEIGHTS,
    HeuristicShelfDetector,
    HybridDetector,
    ObjectDetector,
    YoloDetector,
    _parse_hf_spec,
    build_detector,
    ensure_weights,
)
from smart_shelf.cv.models import BoundingBox, Detection, DetectionResult
from smart_shelf.cv.preprocessing import PreparedImage, prepare_image

pytestmark = pytest.mark.unit


def test_heuristic_detector_satisfies_the_protocol() -> None:
    assert isinstance(HeuristicShelfDetector(), ObjectDetector)


def test_detects_every_facing_on_a_full_shelf(prepared_image: PreparedImage) -> None:
    result = HeuristicShelfDetector().detect(prepared_image)
    # The synthetic shelf has three rows of six facings.
    assert result.count == 18
    assert result.geometry.row_count == 3
    assert all(len(row) == 6 for row in result.by_shelf_level().values())


def test_detects_fewer_facings_when_slots_are_empty(stockout_image_bytes: bytes) -> None:
    stocked = HeuristicShelfDetector().detect(prepare_image(stockout_image_bytes))
    assert stocked.count == 11  # seven of eighteen slots are empty
    assert stocked.occupancy_ratio() < 0.35


def test_detection_is_deterministic(prepared_image: PreparedImage) -> None:
    detector = HeuristicShelfDetector()
    first = detector.detect(prepared_image)
    second = detector.detect(prepared_image)
    assert [d.box for d in first.detections] == [d.box for d in second.detections]


def test_max_items_caps_the_output(prepared_image: PreparedImage) -> None:
    result = HeuristicShelfDetector(max_items=5).detect(prepared_image)
    assert result.count == 5


def test_blank_image_yields_no_detections() -> None:
    blank = PreparedImage(
        pixels=np.full((200, 200, 3), 210, dtype=np.uint8),
        width=200,
        height=200,
        original_width=200,
        original_height=200,
        scale=1.0,
        sha256="0" * 64,
    )
    result = HeuristicShelfDetector().detect(blank)
    assert result.count == 0
    assert result.mean_confidence == 0.0


def test_confidences_stay_in_the_unit_interval(prepared_image: PreparedImage) -> None:
    result = HeuristicShelfDetector().detect(prepared_image)
    assert all(0.0 <= d.confidence <= 1.0 for d in result.detections)


def test_contiguous_runs_finds_every_span() -> None:
    mask = np.array([False, True, True, False, True, False, False, True], dtype=bool)
    assert HeuristicShelfDetector._contiguous_runs(mask) == [(1, 3), (4, 5), (7, 8)]


def test_build_detector_honours_the_configured_backend(settings: Settings) -> None:
    assert isinstance(build_detector(settings), HeuristicShelfDetector)


def test_build_detector_reports_an_unreadable_weights_path(tmp_path: Path) -> None:
    # A path-like value is never treated as a downloadable asset name, so this
    # fails immediately and offline rather than reaching for the network.
    settings = Settings(
        detector_backend=DetectorBackend.YOLO,
        vlm_backend=VLMBackend.MOCK,
        yolo_weights=str(tmp_path / "missing" / "model.pt"),
    )
    with pytest.raises(DetectionError, match="weights not found"):
        build_detector(settings)


class _FakeBoxes:
    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[int]) -> None:
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.conf = np.array(conf, dtype=np.float32)
        self.cls = np.array(cls, dtype=np.int32)


class _FakePrediction:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeYolo:
    names: ClassVar[dict[int, str]] = {0: "bottle", 1: "box"}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def predict(self, *_: Any, **__: Any) -> list[_FakePrediction]:
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return [
            _FakePrediction(
                _FakeBoxes(
                    [[10, 10, 50, 90], [80, 12, 120, 92], [10, 150, 50, 230]],
                    [0.91, 0.84, 0.66],
                    [0, 1, 0],
                )
            )
        ]


def test_yolo_wrapper_maps_predictions_onto_detections(prepared_image: PreparedImage) -> None:
    result = YoloDetector("fake.pt", model=_FakeYolo()).detect(prepared_image)
    assert result.count == 3
    assert {d.label for d in result.detections} == {"bottle", "box"}
    assert result.geometry.row_count == 2
    assert result.detections[0].confidence == pytest.approx(0.91, abs=1e-5)


def test_yolo_inference_failures_surface_as_detection_errors(
    prepared_image: PreparedImage,
) -> None:
    detector = YoloDetector("fake.pt", model=_FakeYolo(fail=True))
    with pytest.raises(DetectionError, match="YOLO inference failed"):
        detector.detect(prepared_image)


class _GpuTensor:
    """Mimics a CUDA torch tensor: numpy conversion fails until ``.cpu()`` runs."""

    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    def cpu(self) -> Any:
        return self._data


class _GpuBoxes:
    def __init__(self) -> None:
        self.xyxy = _GpuTensor([[10, 10, 50, 90], [80, 12, 120, 92]])
        self.conf = _GpuTensor([0.9, 0.8])
        self.cls = _GpuTensor([0, 1])


class _GpuYolo:
    names: ClassVar[dict[int, str]] = {0: "bottle", 1: "box"}

    def predict(self, *_: Any, **__: Any) -> list[_FakePrediction]:
        return [_FakePrediction(_GpuBoxes())]  # type: ignore[arg-type]


def test_predictions_on_a_gpu_are_copied_to_host_memory(
    prepared_image: PreparedImage,
) -> None:
    # numpy cannot read a CUDA tensor directly; without the host copy this
    # raises TypeError on any machine with a GPU.
    result = YoloDetector("fake.pt", model=_GpuYolo()).detect(prepared_image)

    assert result.count == 2
    assert {d.label for d in result.detections} == {"bottle", "box"}


# -- weight resolution and auto-download ---------------------------------------


def stub_downloader(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Install a fake ``ultralytics.utils.downloads`` so no torch import happens."""
    downloads = types.ModuleType("ultralytics.utils.downloads")
    downloads.attempt_download_asset = handler  # type: ignore[attr-defined]
    utils = types.ModuleType("ultralytics.utils")
    utils.downloads = downloads  # type: ignore[attr-defined]
    root = types.ModuleType("ultralytics")
    root.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", root)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.downloads", downloads)


def test_an_existing_weights_file_is_used_as_given(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    assert ensure_weights(str(weights), tmp_path / "cache") == str(weights)


def test_a_previously_cached_checkpoint_is_reused(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "yolov8n.pt").write_bytes(b"weights")

    assert ensure_weights("yolov8n.pt", cache) == str(cache / "yolov8n.pt")


def test_a_missing_explicit_path_is_never_downloaded(tmp_path: Path) -> None:
    # Guessing an asset name from a path would silently load the wrong model.
    with pytest.raises(DetectionError, match="weights not found"):
        ensure_weights(str(tmp_path / "nope" / "model.pt"), tmp_path / "cache")


def test_a_published_name_is_downloaded_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []

    def fake_download(target: str) -> str:
        requested.append(target)
        Path(target).write_bytes(b"downloaded weights")
        return target

    stub_downloader(monkeypatch, fake_download)
    cache = tmp_path / "cache"

    resolved = ensure_weights("yolov8n.pt", cache)

    assert resolved == str(cache / "yolov8n.pt")
    assert requested == [str(cache / "yolov8n.pt")]
    assert (cache / "yolov8n.pt").read_bytes() == b"downloaded weights"


def test_the_cache_directory_is_created_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_downloader(monkeypatch, lambda target: Path(target).write_bytes(b"w"))
    cache = tmp_path / "deeply" / "nested" / "cache"

    ensure_weights("yolov8n.pt", cache)

    assert cache.is_dir()


def test_a_failed_download_explains_how_to_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(target: str) -> str:
        raise OSError("network is unreachable")

    stub_downloader(monkeypatch, offline)

    with pytest.raises(DetectionError, match="could not download") as caught:
        ensure_weights("yolov8n.pt", tmp_path / "cache")

    details = caught.value.details
    assert "network is unreachable" in details["cause"]
    assert "heuristic" in details["hint"]


def test_a_download_that_produces_no_file_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_downloader(monkeypatch, lambda target: target)  # claims success, writes nothing

    with pytest.raises(DetectionError, match="produced no file"):
        ensure_weights("yolov8n.pt", tmp_path / "cache")


def test_the_detector_resolves_weights_through_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "yolov8n.pt").write_bytes(b"weights")
    loaded: list[str] = []

    class _Loader:
        names: ClassVar[dict[int, str]] = {}

        def __init__(self, path: str) -> None:
            loaded.append(path)

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        types.SimpleNamespace(YOLO=_Loader),  # type: ignore[arg-type]
    )
    YoloDetector("yolov8n.pt", weights_dir=cache)

    assert loaded == [str(cache / "yolov8n.pt")]


# -- hybrid backend ------------------------------------------------------------


class _StubDetector:
    """Returns a fixed number of detections, or raises."""

    def __init__(self, count: int, *, name: str = "stub", fail: bool = False) -> None:
        self.name = name
        self._count = count
        self._fail = fail
        self.calls = 0

    def detect(self, image: PreparedImage) -> DetectionResult:
        self.calls += 1
        if self._fail:
            raise DetectionError("backend unavailable")
        return DetectionResult(
            detections=[
                Detection(
                    box=BoundingBox(x1=float(i), y1=0, x2=float(i + 5), y2=10),
                    label="product",
                    confidence=0.9,
                    shelf_level=0,
                    facing_index=i,
                )
                for i in range(self._count)
            ],
            image_width=image.width,
            image_height=image.height,
            backend=self.name,
            duration_ms=1.0,
        )


def test_hybrid_keeps_a_confident_primary_result(prepared_image: PreparedImage) -> None:
    primary, fallback = _StubDetector(40, name="yolo"), _StubDetector(5, name="cv")
    result = HybridDetector(primary, fallback, min_detections=4).detect(prepared_image)

    assert result.count == 40
    assert fallback.calls == 0  # no need to pay for the second pass


def test_hybrid_consults_the_fallback_when_the_primary_is_thin(
    prepared_image: PreparedImage,
) -> None:
    primary, fallback = _StubDetector(1, name="yolo"), _StubDetector(17, name="cv")
    result = HybridDetector(primary, fallback, min_detections=4).detect(prepared_image)

    assert result.count == 17
    assert result.backend == "cv"


def test_hybrid_keeps_the_primary_when_the_fallback_is_worse(
    prepared_image: PreparedImage,
) -> None:
    primary, fallback = _StubDetector(3, name="yolo"), _StubDetector(0, name="cv")
    result = HybridDetector(primary, fallback, min_detections=4).detect(prepared_image)

    assert result.count == 3
    assert result.backend == "yolo"


def test_hybrid_survives_a_primary_that_raises(prepared_image: PreparedImage) -> None:
    primary = _StubDetector(0, name="yolo", fail=True)
    fallback = _StubDetector(12, name="cv")

    result = HybridDetector(primary, fallback, min_detections=4).detect(prepared_image)

    assert result.count == 12
    assert result.backend == "cv"


def test_hybrid_names_both_backends() -> None:
    detector = HybridDetector(_StubDetector(1, name="a"), _StubDetector(1, name="b"))
    assert detector.name == "hybrid:a+b"


def test_hybrid_satisfies_the_protocol() -> None:
    assert isinstance(HybridDetector(_StubDetector(1), _StubDetector(1)), ObjectDetector)


def test_build_detector_falls_back_when_yolo_weights_are_unavailable(
    tmp_path: Path,
) -> None:
    settings = Settings(
        detector_backend=DetectorBackend.HYBRID,
        vlm_backend=VLMBackend.MOCK,
        yolo_weights=str(tmp_path / "missing" / "model.pt"),
    )
    # Unavailable weights must degrade to OpenCV, never refuse to build.
    assert isinstance(build_detector(settings), HeuristicShelfDetector)


# -- Hugging Face weight specs -------------------------------------------------


def test_hf_spec_is_split_into_repo_and_path() -> None:
    repo, path = _parse_hf_spec("hf:owner/repo/weights/model.pt")
    assert repo == "owner/repo"
    assert path == "weights/model.pt"


def test_a_malformed_hf_spec_is_rejected() -> None:
    with pytest.raises(DetectionError, match="malformed"):
        _parse_hf_spec("hf:owner/repo")


def test_the_default_weights_point_at_a_sku110k_checkpoint() -> None:
    # COCO checkpoints detect nothing on shelves, so the default must not be one.
    assert DEFAULT_YOLO_WEIGHTS.startswith("hf:")
    assert "sku110k" in DEFAULT_YOLO_WEIGHTS.lower()


def test_an_hf_checkpoint_is_cached_under_its_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.pt").write_bytes(b"weights")

    # Already cached, so no download is attempted.
    assert ensure_weights("hf:owner/repo/weights/model.pt", cache) == str(cache / "model.pt")


def test_an_hf_checkpoint_downloads_to_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import smart_shelf.cv.detector as module

    def fake_download(weights: str, target: Path) -> None:
        target.write_bytes(b"downloaded")

    monkeypatch.setattr(module, "_download_from_hub", fake_download)
    cache = tmp_path / "cache"

    resolved = ensure_weights("hf:owner/repo/weights/model.pt", cache)

    assert resolved == str(cache / "model.pt")
    assert (cache / "model.pt").read_bytes() == b"downloaded"
