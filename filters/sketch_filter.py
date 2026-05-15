"""Pencil-sketch filter using the classic dodge-blend technique."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class SketchFilter(BaseFilter):
    """Convert the image into a hand-drawn pencil sketch.

    The dodge-blend recipe is:

        gray   = grayscale(image)
        inv    = 255 - gray
        blur   = GaussianBlur(inv, sigma)
        sketch = (gray * 256) / (256 - blur)

    which mimics the photographic *colour dodge* operation. A larger
    blur sigma yields a softer, more pencil-like result.
    """

    name: str = "스케치 효과"
    description: str = (
        "회색조와 반전·블러 이미지의 컬러 닷지 합성을 통해 연필 스케치 같은 "
        "효과를 만듭니다."
    )
    order: int = 330

    def render_controls(self) -> dict[str, Any]:
        sigma = st.sidebar.slider(
            "블러 반경 (σ)",
            min_value=1.0,
            max_value=30.0,
            value=12.0,
            step=0.5,
            help="값이 클수록 더 부드럽고 연필같은 결과가 됩니다.",
            key="sketch_sigma",
        )
        keep_color = st.sidebar.checkbox(
            "컬러 스케치로 유지",
            value=False,
            help="체크 시 원본의 색을 약하게 입혀 컬러 색연필 느낌으로 출력합니다.",
            key="sketch_keep_color",
        )
        return {
            "sigma": float(sigma),
            "keep_color": bool(keep_color),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        sigma: float = 12.0,
        keep_color: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, ksize=(0, 0), sigmaX=max(0.1, sigma))

        # Dodge blend: divide gray by inverted blurred (after another invert)
        # cv2.divide handles the saturation casting back to uint8.
        sketch_gray = cv2.divide(gray, 255 - blurred, scale=256.0)

        if not keep_color:
            return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)

        # Tint the sketch by multiplying with the original colours.
        sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        return np.clip(bgr.astype(np.float32) * sketch_bgr, 0, 255).astype(np.uint8)
