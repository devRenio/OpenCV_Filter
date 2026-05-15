"""Color isolation / color splash filter.

The user picks a hue range; the filter keeps that range in colour and
desaturates everything else. A soft mask is used so the boundary
between coloured and grey regions does not look harsh.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

# Common hue presets in OpenCV's 0..179 hue scale. The dictionary preserves
# insertion order so it doubles as the selectbox option list.
_HUE_PRESETS: dict[str, int] = {
    "빨강": 0,
    "주황": 15,
    "노랑": 30,
    "초록": 60,
    "청록": 90,
    "파랑": 110,
    "보라": 140,
    "분홍": 165,
    "직접 지정": -1,
}


@register
class ColorIsolationFilter(BaseFilter):
    """Keep a single hue range coloured; convert the rest to grayscale."""

    name: str = "특정 색상 강조"
    description: str = (
        "지정한 색상 범위만 컬러로 유지하고 나머지 영역은 흑백으로 처리해 강조 "
        "효과(컬러 스플래시)를 만듭니다."
    )
    order: int = 140

    def render_controls(self) -> dict[str, Any]:
        preset_label = st.sidebar.selectbox(
            "강조할 색상",
            options=list(_HUE_PRESETS.keys()),
            index=0,
            key="color_iso_preset",
        )
        preset_value = _HUE_PRESETS[preset_label]

        hue_center = st.sidebar.slider(
            "Hue 중심 (0–179)",
            min_value=0,
            max_value=179,
            value=int(preset_value if preset_value >= 0 else 0),
            step=1,
            help="OpenCV의 HSV에서 H 값은 0~179 범위입니다.",
            key="color_iso_hue",
            disabled=preset_value >= 0,
        )
        hue_width = st.sidebar.slider(
            "Hue 허용 범위 (±)",
            min_value=1,
            max_value=60,
            value=15,
            step=1,
            help="중심에서 좌우로 몇 만큼의 색상까지 허용할지 결정합니다.",
            key="color_iso_width",
        )
        sat_min = st.sidebar.slider(
            "최소 채도",
            min_value=0,
            max_value=255,
            value=80,
            step=1,
            help="이 값보다 채도가 낮은 픽셀(거의 회색)은 제외합니다.",
            key="color_iso_sat",
        )
        feather = st.sidebar.slider(
            "마스크 부드러움",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="마스크 경계를 가우시안 블러로 부드럽게 합니다.",
            key="color_iso_feather",
        )
        center = preset_value if preset_value >= 0 else hue_center
        return {
            "hue_center": int(center),
            "hue_width": int(hue_width),
            "sat_min": int(sat_min),
            "feather": float(feather),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        hue_center: int = 0,
        hue_width: int = 15,
        sat_min: int = 80,
        feather: float = 2.0,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask = self._build_hue_mask(hsv, hue_center, hue_width, sat_min)

        if feather > 0:
            mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=float(feather))

        # Composite: keep colour where mask is high, grayscale elsewhere.
        gray_bgr = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        blended = bgr.astype(np.float32) * alpha + gray_bgr.astype(np.float32) * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _build_hue_mask(
        hsv: np.ndarray,
        hue_center: int,
        hue_width: int,
        sat_min: int,
    ) -> np.ndarray:
        """Return a uint8 mask (0 / 255) for the requested HSV range.

        The hue axis is circular (0 == 180 in OpenCV's 0..179 scale), so we
        OR two ranges when the requested window wraps around 0.
        """
        low = (hue_center - hue_width) % 180
        high = (hue_center + hue_width) % 180

        sat_high = np.array([0, 255, 255], dtype=np.uint8)
        sat_low = np.array([0, sat_min, 30], dtype=np.uint8)

        if low <= high:
            lower = sat_low.copy()
            upper = sat_high.copy()
            lower[0] = low
            upper[0] = high
            return cv2.inRange(hsv, lower, upper)

        # Wrap-around case: union of [low..179] and [0..high]
        lower_a = sat_low.copy()
        upper_a = sat_high.copy()
        lower_a[0] = low
        upper_a[0] = 179

        lower_b = sat_low.copy()
        upper_b = sat_high.copy()
        lower_b[0] = 0
        upper_b[0] = high

        return cv2.bitwise_or(
            cv2.inRange(hsv, lower_a, upper_a),
            cv2.inRange(hsv, lower_b, upper_b),
        )
