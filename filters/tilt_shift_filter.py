"""Tilt-shift / miniature effect filter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class TiltShiftFilter(BaseFilter):
    """Create a fake tilt-shift effect that makes scenes look like miniatures.

    The effect works by:
    1. Keeping a horizontal band in focus (sharp)
    2. Progressively blurring the regions above and below
    3. Optionally boosting saturation for a toy-like appearance
    """

    name: str = "틸트 시프트 (미니어처)"
    description: str = (
        "중앙 영역만 선명하게 두고 상하를 강하게 블러 처리해 미니어처/장난감 "
        "세계처럼 보이게 만듭니다."
    )
    order: int = 510

    def render_controls(self) -> dict[str, Any]:
        focus_position = st.sidebar.slider(
            "초점 위치 (%)",
            min_value=10,
            max_value=90,
            value=50,
            step=1,
            help="선명한 영역의 수직 위치. 50이 중앙입니다.",
            key="tilt_shift_position",
        )
        focus_width = st.sidebar.slider(
            "초점 영역 폭 (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            help="선명하게 유지할 영역의 높이 비율",
            key="tilt_shift_width",
        )
        blur_strength = st.sidebar.slider(
            "블러 강도",
            min_value=5,
            max_value=50,
            value=25,
            step=1,
            key="tilt_shift_blur",
        )
        saturation_boost = st.sidebar.slider(
            "채도 증가",
            min_value=1.0,
            max_value=2.0,
            value=1.3,
            step=0.05,
            help="색감을 진하게 해서 장난감 느낌을 강조합니다.",
            key="tilt_shift_saturation",
        )
        return {
            "focus_position": int(focus_position),
            "focus_width": int(focus_width),
            "blur_strength": int(blur_strength),
            "saturation_boost": float(saturation_boost),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        focus_position: int = 50,
        focus_width: int = 20,
        blur_strength: int = 25,
        saturation_boost: float = 1.3,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        # Calculate focus band boundaries
        center_y = int(h * focus_position / 100.0)
        half_width = int(h * focus_width / 100.0 / 2.0)
        focus_top = max(0, center_y - half_width)
        focus_bottom = min(h, center_y + half_width)

        # Create heavily blurred version
        ksize = blur_strength | 1  # ensure odd
        blurred = cv2.GaussianBlur(bgr, (ksize, ksize), 0)

        # Build gradient mask for smooth transition
        mask = self._build_gradient_mask(h, w, focus_top, focus_bottom)

        # Blend sharp center with blurred edges
        mask_3ch = mask[:, :, np.newaxis]
        result = (bgr.astype(np.float32) * mask_3ch +
                  blurred.astype(np.float32) * (1.0 - mask_3ch))
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Boost saturation for toy-like appearance
        if saturation_boost > 1.0:
            result = self._boost_saturation(result, saturation_boost)

        return result

    @staticmethod
    def _build_gradient_mask(
        h: int, w: int, focus_top: int, focus_bottom: int
    ) -> np.ndarray:
        """Build a vertical gradient mask: 1.0 in focus band, fading to 0.0."""
        mask = np.zeros((h, w), dtype=np.float32)

        # Focus band is fully sharp (1.0)
        mask[focus_top:focus_bottom, :] = 1.0

        # Gradient above focus band
        if focus_top > 0:
            for y in range(focus_top):
                # Quadratic falloff for more natural DoF simulation
                ratio = (y / focus_top) ** 1.5
                mask[y, :] = ratio

        # Gradient below focus band
        if focus_bottom < h:
            remaining = h - focus_bottom
            for y in range(focus_bottom, h):
                ratio = ((h - 1 - y) / remaining) ** 1.5
                mask[y, :] = ratio

        return mask

    @staticmethod
    def _boost_saturation(bgr: np.ndarray, factor: float) -> np.ndarray:
        """Increase saturation in HSV space."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
