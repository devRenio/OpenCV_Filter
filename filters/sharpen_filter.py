"""Unsharp-mask based sharpen filter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class SharpenFilter(BaseFilter):
    """Sharpen the image by amplifying the difference against a blurred copy.

    This implementation uses the *unsharp mask* technique:

        sharpened = original + strength * (original - blurred)

    which yields more natural results than a plain Laplacian convolution.
    """

    name: str = "샤프닝"
    description: str = "이미지의 엣지를 강조해 더 선명하게 보이도록 합니다."
    order: int = 220

    def render_controls(self) -> dict[str, Any]:
        strength = st.sidebar.slider(
            "강도",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="0이면 원본, 값이 클수록 엣지가 강조됩니다.",
            key="sharpen_strength",
        )
        radius = st.sidebar.slider(
            "반경 (σ)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="블러 가우시안의 표준편차. 클수록 더 큰 디테일이 강조됩니다.",
            key="sharpen_radius",
        )
        return {"strength": float(strength), "radius": float(radius)}

    def apply(
        self,
        image: np.ndarray,
        *,
        strength: float = 1.0,
        radius: float = 2.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply unsharp-mask sharpening."""
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if strength <= 0:
            return bgr

        blurred = cv2.GaussianBlur(bgr, ksize=(0, 0), sigmaX=max(0.1, radius))
        sharpened = cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0.0)
        return sharpened
