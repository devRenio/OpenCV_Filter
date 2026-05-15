"""Detect faces with a Haar cascade and pixelate the detected regions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

logger = logging.getLogger(__name__)


@register
class FaceMosaicFilter(BaseFilter):
    """Anonymise people by mosaicing every detected face.

    Implementation notes:

    * The Haar frontal-face cascade ships with the ``opencv-python*``
      wheels under ``cv2.data.haarcascades`` so no external download is
      required.
    * To keep webcam frame rates usable, detection runs on a copy
      resized so the longest side is around ``_DETECT_SIDE`` pixels.
      The detected boxes are then scaled back to original coordinates.
    """

    name: str = "얼굴 모자이크"
    description: str = (
        "Haar Cascade로 얼굴을 검출한 뒤 해당 영역만 모자이크 처리하여 "
        "익명화합니다 (정면 얼굴 한정)."
    )
    order: int = 410

    _DETECT_SIDE: ClassVar[int] = 400
    _CASCADE: ClassVar[cv2.CascadeClassifier | None] = None

    @classmethod
    def _get_cascade(cls) -> cv2.CascadeClassifier | None:
        """Lazily load (and cache) the bundled frontal-face cascade."""
        if cls._CASCADE is not None:
            return cls._CASCADE

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            logger.error("Haar cascade XML not found at %s", cascade_path)
            return None

        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            logger.error("Failed to load Haar cascade from %s", cascade_path)
            return None

        cls._CASCADE = cascade
        return cascade

    def render_controls(self) -> dict[str, Any]:
        pixel_size = st.sidebar.slider(
            "모자이크 픽셀 크기",
            min_value=4,
            max_value=60,
            value=18,
            step=1,
            key="face_mosaic_pixel_size",
        )
        scale_factor = st.sidebar.slider(
            "scaleFactor",
            min_value=1.05,
            max_value=1.6,
            value=1.2,
            step=0.05,
            help="검출 피라미드 단계 비율. 작을수록 더 꼼꼼히 탐색하지만 느려집니다.",
            key="face_mosaic_scale",
        )
        min_neighbors = st.sidebar.slider(
            "minNeighbors",
            min_value=1,
            max_value=12,
            value=5,
            step=1,
            help="검출 후보가 이웃에서 몇 번 이상 동의해야 얼굴로 인정할지.",
            key="face_mosaic_min_neighbors",
        )
        padding = st.sidebar.slider(
            "검출 박스 여백 (%)",
            min_value=0,
            max_value=50,
            value=15,
            step=1,
            help="얼굴 박스를 가로/세로로 몇 % 확장해 모자이크할지.",
            key="face_mosaic_padding",
        )
        return {
            "pixel_size": int(pixel_size),
            "scale_factor": float(scale_factor),
            "min_neighbors": int(min_neighbors),
            "padding": int(padding),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        pixel_size: int = 18,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        padding: int = 15,
        **kwargs: Any,
    ) -> np.ndarray:
        cascade = self._get_cascade()
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if cascade is None:
            return bgr

        h, w = bgr.shape[:2]
        scale = max(1.0, max(h, w) / float(self._DETECT_SIDE))
        if scale > 1.0:
            small = cv2.resize(bgr, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
        else:
            small = bgr

        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.equalizeHist(gray_small)

        faces = cascade.detectMultiScale(
            gray_small,
            scaleFactor=max(1.05, scale_factor),
            minNeighbors=max(1, min_neighbors),
            minSize=(24, 24),
        )

        if len(faces) == 0:
            return bgr

        result = bgr.copy()
        pad_ratio = max(0.0, padding / 100.0)
        block = max(2, int(pixel_size))

        for (fx, fy, fw, fh) in faces:
            # Scale detection coordinates back to the original image.
            x = int(fx * scale)
            y = int(fy * scale)
            ww = int(fw * scale)
            hh = int(fh * scale)

            # Pad by 'padding' percent.
            dx = int(ww * pad_ratio)
            dy = int(hh * pad_ratio)
            x0 = max(0, x - dx)
            y0 = max(0, y - dy)
            x1 = min(w, x + ww + dx)
            y1 = min(h, y + hh + dy)

            roi = result[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            roi_h, roi_w = roi.shape[:2]
            nw = max(1, roi_w // block)
            nh = max(1, roi_h // block)
            small_roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
            result[y0:y1, x0:x1] = cv2.resize(small_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        return result
