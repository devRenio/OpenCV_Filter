"""Face mosaic / anonymisation filter with pluggable detection backends.

This module implements the Strategy pattern for face detection so that
the user can experiment with several algorithms behind a single filter
entry in the UI:

* **Haar Cascade (정면, 빠름)** - the classic Viola-Jones detector
  bundled with OpenCV. Fast but brittle on non-frontal faces.

* **Haar Cascade (정면+측면 결합)** - combines four cascades
  (frontal default + frontal alt2 + profile + horizontally flipped
  profile) and merges the detections with :func:`cv2.dnn.NMSBoxes` to
  cover side views without any model download.

* **DNN (Caffe SSD)** - OpenCV's res10 300x300 SSD detector
  (Aleksandr Rybnikov, 2017). Significantly more robust than Haar at a
  modest runtime cost. Model weights (~10 MB) are auto-downloaded on
  first use.

* **YuNet (최신, ONNX)** - 2023's `face_detection_yunet` from
  ``opencv_zoo``: tiny (~340 KB) and currently the recommended
  built-in CNN face detector in OpenCV. Returns per-face confidence
  scores natively.

The obscuration step is also pluggable: mosaic / Gaussian blur /
solid colour box. The result is always a 3-channel BGR image so the
rest of the pipeline can treat it uniformly.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Final

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

logger = logging.getLogger(__name__)

# All downloaded model artefacts live alongside the source tree so they
# survive across sessions but stay out of version control (see .gitignore).
_MODELS_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "models"


def _ensure_model(filename: str, url: str) -> Path:
    """Download ``url`` into :data:`_MODELS_DIR` if not already cached.

    The download is performed atomically via a ``.part`` temp file so a
    partial transfer cannot leave a corrupt cache behind.

    Args:
        filename: Destination filename inside ``models/``.
        url: HTTPS URL to fetch.

    Returns:
        The path to the cached file on success.

    Raises:
        RuntimeError: If the download fails for any reason.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = _MODELS_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        return target

    tmp_path = target.with_suffix(target.suffix + ".part")
    logger.info("Downloading face detection model %s from %s", filename, url)
    with st.spinner(f"필터 모델 다운로드 중: {filename} (최초 1회만 수행)"):
        try:
            urllib.request.urlretrieve(url, str(tmp_path))
            tmp_path.replace(target)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"모델 파일 다운로드에 실패했습니다: {filename}. "
                f"네트워크 연결을 확인해 주세요. 원본 오류: {exc}"
            ) from exc
    return target


# ---------------------------------------------------------------------------
# Detector strategy interface and implementations
# ---------------------------------------------------------------------------
class _BaseFaceDetector(ABC):
    """Strategy interface implemented by each detection backend.

    Subclasses must define the class attributes :attr:`label` and
    :attr:`supports_confidence`, and implement :meth:`detect`.

    Instances are cached per-subclass via :meth:`get` so the heavy
    one-time work (loading a network, parsing a cascade XML) only
    happens once per detector type across the whole Streamlit session.
    """

    label: ClassVar[str] = ""
    supports_confidence: ClassVar[bool] = False

    _instances: ClassVar[dict[type, "_BaseFaceDetector"]] = {}

    @classmethod
    def get(cls) -> "_BaseFaceDetector":
        """Return a process-wide singleton instance of the subclass."""
        if cls not in _BaseFaceDetector._instances:
            _BaseFaceDetector._instances[cls] = cls()
        return _BaseFaceDetector._instances[cls]

    @abstractmethod
    def detect(
        self,
        bgr: np.ndarray,
        *,
        confidence_threshold: float = 0.5,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
    ) -> list[tuple[int, int, int, int, float]]:
        """Detect faces in ``bgr``.

        Returns:
            A list of ``(x, y, w, h, confidence)`` tuples in pixel
            coordinates of the *input* image. ``confidence`` is in
            ``[0, 1]``; detectors that do not produce confidence scores
            should return ``1.0``.
        """


class _HaarFrontalDetector(_BaseFaceDetector):
    """Single frontal-face Viola-Jones cascade."""

    label = "Haar Cascade (정면, 빠름)"
    supports_confidence = False

    def __init__(self) -> None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise RuntimeError(f"Haar cascade를 불러오지 못했습니다: {path}")

    def detect(
        self,
        bgr: np.ndarray,
        *,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        **_: Any,
    ) -> list[tuple[int, int, int, int, float]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=max(1.05, scale_factor),
            minNeighbors=max(1, min_neighbors),
            minSize=(24, 24),
        )
        return [(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in rects]


class _HaarCombinedDetector(_BaseFaceDetector):
    """Combine four Haar cascades and dedupe with NMS for broad coverage."""

    label = "Haar Cascade (정면+측면 결합)"
    supports_confidence = False

    def __init__(self) -> None:
        haar_dir = Path(cv2.data.haarcascades)
        self._cascades: list[tuple[str, cv2.CascadeClassifier]] = []
        for name in (
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        ):
            cascade = cv2.CascadeClassifier(str(haar_dir / name))
            if cascade.empty():
                raise RuntimeError(f"Haar cascade를 불러오지 못했습니다: {name}")
            self._cascades.append((name, cascade))

    def detect(
        self,
        bgr: np.ndarray,
        *,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        **_: Any,
    ) -> list[tuple[int, int, int, int, float]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        sf = max(1.05, scale_factor)
        mn = max(1, min_neighbors)

        all_rects: list[tuple[int, int, int, int]] = []
        for name, cascade in self._cascades:
            rects = cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(24, 24))
            all_rects.extend((int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects)

        # Right-facing profile: opencv's profileface cascade is trained for
        # one side only, so flip the image and re-run it.
        h, w = gray.shape
        flipped = cv2.flip(gray, 1)
        profile_cascade = next(c for name, c in self._cascades if "profileface" in name)
        flipped_rects = profile_cascade.detectMultiScale(
            flipped, scaleFactor=sf, minNeighbors=mn, minSize=(24, 24)
        )
        for (x, y, fw, fh) in flipped_rects:
            all_rects.append((int(w - x - fw), int(y), int(fw), int(fh)))

        if not all_rects:
            return []

        boxes = [list(r) for r in all_rects]
        scores = [1.0] * len(boxes)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.3)
        if indices is None or len(indices) == 0:
            return []

        # cv2.dnn.NMSBoxes returns an array shape (N,) on modern opencv
        # and (N, 1) on older builds; normalise to a flat iterable.
        flat = np.array(indices).flatten().tolist()
        return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], 1.0) for i in flat]


class _DnnSsdDetector(_BaseFaceDetector):
    """Caffe-based SSD detector with a ResNet-10 backbone (res10 300x300)."""

    label = "DNN (Caffe SSD)"
    supports_confidence = True

    _PROTOTXT_FILE = "deploy.prototxt"
    _MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
    _PROTOTXT_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    )
    _MODEL_URL = (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    )

    def __init__(self) -> None:
        proto = _ensure_model(self._PROTOTXT_FILE, self._PROTOTXT_URL)
        model = _ensure_model(self._MODEL_FILE, self._MODEL_URL)
        self._net = cv2.dnn.readNetFromCaffe(str(proto), str(model))

    def detect(
        self,
        bgr: np.ndarray,
        *,
        confidence_threshold: float = 0.5,
        **_: Any,
    ) -> list[tuple[int, int, int, int, float]]:
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        detections = self._net.forward()  # shape (1, 1, N, 7)

        results: list[tuple[int, int, int, int, float]] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), conf))
        return results


class _YuNetDetector(_BaseFaceDetector):
    """Modern YuNet face detector exposed via ``cv2.FaceDetectorYN``."""

    label = "YuNet (최신, ONNX)"
    supports_confidence = True

    _MODEL_FILE = "face_detection_yunet_2023mar.onnx"
    _MODEL_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )

    def __init__(self) -> None:
        model = _ensure_model(self._MODEL_FILE, self._MODEL_URL)
        self._detector = cv2.FaceDetectorYN.create(
            model=str(model),
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(
        self,
        bgr: np.ndarray,
        *,
        confidence_threshold: float = 0.5,
        **_: Any,
    ) -> list[tuple[int, int, int, int, float]]:
        h, w = bgr.shape[:2]
        self._detector.setInputSize((w, h))
        self._detector.setScoreThreshold(float(confidence_threshold))

        _, faces = self._detector.detect(bgr)
        if faces is None:
            return []

        results: list[tuple[int, int, int, int, float]] = []
        for face in faces:
            x, y, fw, fh = face[0:4]
            conf = float(face[-1])
            results.append((int(max(0, x)), int(max(0, y)), int(fw), int(fh), conf))
        return results


_DETECTORS: Final[dict[str, type[_BaseFaceDetector]]] = {
    _HaarFrontalDetector.label: _HaarFrontalDetector,
    _HaarCombinedDetector.label: _HaarCombinedDetector,
    _DnnSsdDetector.label: _DnnSsdDetector,
    _YuNetDetector.label: _YuNetDetector,
}

_OBSCURE_METHODS: Final[dict[str, str]] = {
    "모자이크 (픽셀화)": "mosaic",
    "가우시안 블러": "blur",
    "검은 박스": "blackbox",
    "흰 박스": "whitebox",
}


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
@register
class FaceMosaicFilter(BaseFilter):
    """Detect faces with a user-selected backend and obscure each region."""

    name: str = "얼굴 모자이크"
    description: str = (
        "다양한 얼굴 검출 알고리즘(Haar / DNN / YuNet)을 골라 얼굴을 찾고, "
        "원하는 방식(모자이크 / 가우시안 블러 / 색 박스)으로 가려 익명화합니다."
    )
    order: int = 410

    _DETECT_SIDE: ClassVar[int] = 480  # downsample target for detection

    def render_controls(self) -> dict[str, Any]:
        detector_label = st.sidebar.selectbox(
            "검출 알고리즘",
            options=list(_DETECTORS.keys()),
            index=0,
            help="DNN과 YuNet은 첫 사용 시 모델 파일을 자동 다운로드합니다.",
            key="face_mosaic_detector",
        )
        detector_cls = _DETECTORS[detector_label]
        is_dnn = detector_cls.supports_confidence

        confidence = st.sidebar.slider(
            "신뢰도 임계값",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            disabled=not is_dnn,
            help="DNN / YuNet 전용. Haar 방식에서는 이 값이 무시됩니다.",
            key="face_mosaic_confidence",
        )

        scale_factor = st.sidebar.slider(
            "scaleFactor (Haar 전용)",
            min_value=1.05,
            max_value=1.6,
            value=1.2,
            step=0.05,
            disabled=is_dnn,
            help="검출 피라미드의 스케일 비율. 작을수록 더 꼼꼼히 탐색하지만 느려집니다.",
            key="face_mosaic_scale",
        )
        min_neighbors = st.sidebar.slider(
            "minNeighbors (Haar 전용)",
            min_value=1,
            max_value=12,
            value=5,
            step=1,
            disabled=is_dnn,
            help="검출 후보가 이웃에서 몇 번 이상 동의해야 얼굴로 인정할지 결정합니다.",
            key="face_mosaic_min_neighbors",
        )

        obscure_label = st.sidebar.selectbox(
            "가림 방식",
            options=list(_OBSCURE_METHODS.keys()),
            index=0,
            key="face_mosaic_obscure",
        )
        obscure_method = _OBSCURE_METHODS[obscure_label]

        if obscure_method == "mosaic":
            strength = float(
                st.sidebar.slider(
                    "모자이크 픽셀 크기",
                    min_value=4,
                    max_value=60,
                    value=18,
                    step=1,
                    key="face_mosaic_strength_mosaic",
                )
            )
        elif obscure_method == "blur":
            strength = float(
                st.sidebar.slider(
                    "블러 강도 (σ)",
                    min_value=1.0,
                    max_value=40.0,
                    value=15.0,
                    step=0.5,
                    key="face_mosaic_strength_blur",
                )
            )
        else:
            strength = 0.0

        padding = st.sidebar.slider(
            "박스 여백 (%)",
            min_value=0,
            max_value=60,
            value=15,
            step=1,
            help="검출 박스를 상하좌우로 몇 % 확장해 가릴지 결정합니다.",
            key="face_mosaic_padding",
        )
        show_boxes = st.sidebar.checkbox(
            "검출 박스 표시 (디버그)",
            value=False,
            help="가린 영역 주변에 박스와 신뢰도를 함께 표시합니다.",
            key="face_mosaic_show_boxes",
        )

        return {
            "detector_label": detector_label,
            "confidence_threshold": float(confidence),
            "scale_factor": float(scale_factor),
            "min_neighbors": int(min_neighbors),
            "obscure_method": obscure_method,
            "strength": float(strength),
            "padding": int(padding),
            "show_boxes": bool(show_boxes),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        detector_label: str = _HaarFrontalDetector.label,
        confidence_threshold: float = 0.5,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        obscure_method: str = "mosaic",
        strength: float = 18.0,
        padding: int = 15,
        show_boxes: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        detector_cls = _DETECTORS.get(detector_label, _HaarFrontalDetector)
        try:
            detector = detector_cls.get()
        except Exception as exc:  # noqa: BLE001 - surface init errors to UI
            logger.exception("Failed to initialise detector %s", detector_label)
            st.warning(f"'{detector_label}' 초기화 실패: {exc}")
            return bgr

        h, w = bgr.shape[:2]
        scale = max(1.0, max(h, w) / float(self._DETECT_SIDE))
        if scale > 1.0:
            small = cv2.resize(bgr, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
        else:
            small = bgr

        try:
            detections = detector.detect(
                small,
                confidence_threshold=confidence_threshold,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
            )
        except Exception as exc:  # noqa: BLE001 - filters are user-supplied plug-ins
            logger.exception("Detection failed in %s", detector_label)
            st.warning(f"얼굴 검출 중 오류가 발생했습니다: {exc}")
            return bgr

        if not detections:
            return bgr

        result = bgr.copy()
        pad_ratio = max(0.0, padding / 100.0)

        for (fx, fy, fw, fh, conf) in detections:
            # Scale detection coordinates back to the original image.
            x = int(fx * scale)
            y = int(fy * scale)
            ww = int(fw * scale)
            hh = int(fh * scale)

            dx = int(ww * pad_ratio)
            dy = int(hh * pad_ratio)
            x0 = max(0, x - dx)
            y0 = max(0, y - dy)
            x1 = min(w, x + ww + dx)
            y1 = min(h, y + hh + dy)

            roi = result[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            result[y0:y1, x0:x1] = self._obscure_roi(roi, obscure_method, strength)

            if show_boxes:
                self._draw_box(result, x0, y0, x1, y1, conf, detector_cls.supports_confidence)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _obscure_roi(roi: np.ndarray, method: str, strength: float) -> np.ndarray:
        """Return a copy of ``roi`` obscured by the chosen method."""
        rh, rw = roi.shape[:2]
        if method == "mosaic":
            block = max(2, int(strength) if strength > 0 else 18)
            nw = max(1, rw // block)
            nh = max(1, rh // block)
            small = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        if method == "blur":
            sigma = max(1.0, float(strength))
            return cv2.GaussianBlur(roi, ksize=(0, 0), sigmaX=sigma)
        if method == "blackbox":
            return np.zeros_like(roi)
        if method == "whitebox":
            return np.full_like(roi, 255)
        return roi

    @staticmethod
    def _draw_box(
        canvas: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        confidence: float,
        show_confidence: bool,
    ) -> None:
        """Overlay a green bounding box (and optional confidence label)."""
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)
        if not show_confidence:
            return

        label = f"{confidence:.2f}"
        text_origin = (x0, max(15, y0 - 6))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, label, text_origin, font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, text_origin, font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
