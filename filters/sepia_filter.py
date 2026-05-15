"""Sepia tone filter with adjustable blend strength."""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

# Classic sepia matrix expressed for *BGR* input/output. It is the standard
# RGB sepia matrix with both axes flipped so the math stays in OpenCV's
# native channel order.
_SEPIA_BGR_MATRIX: Final[np.ndarray] = np.array(
    [
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ],
    dtype=np.float32,
)


@register
class SepiaFilter(BaseFilter):
    """Apply a warm brown sepia tone reminiscent of classic photographs."""

    name: str = "세피아"
    description: str = "고전 사진 느낌의 따뜻한 갈색 톤으로 변환합니다."
    order: int = 130

    def render_controls(self) -> dict[str, Any]:
        intensity = st.sidebar.slider(
            "강도",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="0이면 원본, 1이면 완전한 세피아 적용입니다.",
            key="sepia_intensity",
        )
        return {"intensity": float(intensity)}

    def apply(
        self,
        image: np.ndarray,
        *,
        intensity: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Blend the sepia-transformed image with the original."""
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        sepia = cv2.transform(bgr, _SEPIA_BGR_MATRIX)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)

        if intensity >= 1.0:
            return sepia
        if intensity <= 0.0:
            return bgr
        return cv2.addWeighted(bgr, 1.0 - intensity, sepia, intensity, 0.0)
