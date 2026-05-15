"""Gaussian blur filter with adjustable kernel size and sigma."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class GaussianBlurFilter(BaseFilter):
    """Smooth the image with a Gaussian kernel."""

    name: str = "가우시안 블러"
    description: str = "가우시안 커널로 이미지를 부드럽게 블러 처리합니다."
    order: int = 200

    def render_controls(self) -> dict[str, Any]:
        kernel_size = st.sidebar.slider(
            "커널 크기",
            min_value=1,
            max_value=31,
            value=5,
            step=2,
            help="커널 한 변의 픽셀 수. 홀수만 허용됩니다.",
            key="gaussian_kernel",
        )
        sigma = st.sidebar.slider(
            "표준편차 (σ)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="0으로 두면 커널 크기로부터 자동 계산됩니다.",
            key="gaussian_sigma",
        )
        return {"kernel_size": int(kernel_size), "sigma": float(sigma)}

    def apply(
        self,
        image: np.ndarray,
        *,
        kernel_size: int = 5,
        sigma: float = 0.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply :func:`cv2.GaussianBlur` and return a 3-channel BGR image."""
        k = max(1, kernel_size | 1)
        blurred = cv2.GaussianBlur(image, (k, k), sigmaX=float(sigma))
        if blurred.ndim == 2:
            return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return blurred
