"""Pixelate / mosaic filter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class PixelateFilter(BaseFilter):
    """Reduce the image to coarse pixel blocks for a mosaic effect."""

    name: str = "모자이크 (픽셀화)"
    description: str = "이미지를 작은 블록 단위로 단순화해 모자이크 효과를 만듭니다."
    order: int = 400

    def render_controls(self) -> dict[str, Any]:
        pixel_size = st.sidebar.slider(
            "픽셀 크기",
            min_value=2,
            max_value=50,
            value=10,
            step=1,
            help="값이 클수록 한 블록의 크기가 커지고 더 거친 모자이크가 만들어집니다.",
            key="pixel_size",
        )
        return {"pixel_size": int(pixel_size)}

    def apply(
        self,
        image: np.ndarray,
        *,
        pixel_size: int = 10,
        **kwargs: Any,
    ) -> np.ndarray:
        """Downsample with linear interpolation, then upsample with nearest-neighbour.

        The two-step resize is the classic OpenCV idiom for pixelation:
        the linear downsample averages the block contents and the nearest
        upsample replicates the result to fill each block.
        """
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        block = max(2, int(pixel_size))
        new_w = max(1, w // block)
        new_h = max(1, h // block)

        small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
