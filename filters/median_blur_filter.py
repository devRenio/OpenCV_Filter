"""Median blur filter, especially effective at suppressing salt-and-pepper noise."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class MedianBlurFilter(BaseFilter):
    """Replace each pixel with the median of its neighbourhood."""

    name: str = "미디언 블러"
    description: str = (
        "주변 픽셀의 중앙값으로 점 잡음(salt-and-pepper noise)을 효과적으로 제거합니다."
    )
    order: int = 210

    def render_controls(self) -> dict[str, Any]:
        kernel_size = st.sidebar.slider(
            "커널 크기",
            min_value=1,
            max_value=21,
            value=5,
            step=2,
            help="홀수만 허용됩니다. 값이 클수록 더 강한 평활화 효과가 적용됩니다.",
            key="median_kernel",
        )
        return {"kernel_size": int(kernel_size)}

    def apply(
        self,
        image: np.ndarray,
        *,
        kernel_size: int = 5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply :func:`cv2.medianBlur` and return a 3-channel BGR image."""
        k = max(1, kernel_size | 1)
        blurred = cv2.medianBlur(image, k)
        if blurred.ndim == 2:
            return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return blurred
