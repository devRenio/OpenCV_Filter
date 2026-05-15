"""Canny edge detection filter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class CannyEdgeFilter(BaseFilter):
    """Extract image contours using the Canny edge detector."""

    name: str = "엣지 검출 (Canny)"
    description: str = "Canny 알고리즘으로 이미지의 윤곽선을 추출합니다."
    order: int = 300

    def render_controls(self) -> dict[str, Any]:
        low = st.sidebar.slider(
            "최소 임계값",
            min_value=0,
            max_value=255,
            value=100,
            step=1,
            help="이 값보다 약한 그래디언트는 엣지에서 제외됩니다.",
            key="canny_low",
        )
        high = st.sidebar.slider(
            "최대 임계값",
            min_value=0,
            max_value=255,
            value=200,
            step=1,
            help="이 값보다 강한 그래디언트는 확실한 엣지로 분류됩니다.",
            key="canny_high",
        )
        blur = st.sidebar.slider(
            "사전 블러 (σ)",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="0이면 블러를 생략합니다. 노이즈가 많을 때 값을 키우면 결과가 깔끔해집니다.",
            key="canny_blur",
        )
        return {
            "low": int(low),
            "high": int(high),
            "blur_sigma": float(blur),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        low: int = 100,
        high: int = 200,
        blur_sigma: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run Canny edge detection and return a 3-channel BGR image."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if blur_sigma > 0:
            gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=float(blur_sigma))

        # OpenCV expects threshold1 <= threshold2; swap defensively.
        if low > high:
            low, high = high, low

        edges = cv2.Canny(gray, threshold1=int(low), threshold2=int(high))
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
