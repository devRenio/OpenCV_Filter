"""Binary threshold filter (with optional Otsu auto-threshold)."""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

_METHODS: Final[dict[str, int]] = {
    "이진화": cv2.THRESH_BINARY,
    "이진화 (반전)": cv2.THRESH_BINARY_INV,
    "Otsu (자동 임계값)": cv2.THRESH_BINARY | cv2.THRESH_OTSU,
}


@register
class ThresholdFilter(BaseFilter):
    """Split the image into two intensity levels around a threshold value."""

    name: str = "이진화 (Threshold)"
    description: str = "지정한 임계값을 기준으로 이미지를 흑백 두 단계로 분할합니다."
    order: int = 310

    def render_controls(self) -> dict[str, Any]:
        method_label = st.sidebar.selectbox(
            "방식",
            options=list(_METHODS.keys()),
            index=0,
            help="Otsu는 임계값을 자동으로 추정합니다 (수동 임계값 무시).",
            key="thresh_method",
        )
        threshold = st.sidebar.slider(
            "임계값",
            min_value=0,
            max_value=255,
            value=127,
            step=1,
            help="Otsu 방식에서는 무시됩니다.",
            key="thresh_value",
        )
        return {
            "method_label": method_label,
            "threshold": int(threshold),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        method_label: str = "이진화",
        threshold: int = 127,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply :func:`cv2.threshold` and return a 3-channel BGR image."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flag = _METHODS.get(method_label, cv2.THRESH_BINARY)
        _, binary = cv2.threshold(gray, float(threshold), 255, flag)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
