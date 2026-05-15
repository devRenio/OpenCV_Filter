"""Moody / cinematic black-and-white filter.

Combines a contrast-boosted grayscale conversion, a soft Gaussian
vignette and (optionally) film grain to produce a more emotional B&W
look than a plain ``cv2.cvtColor`` call.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class MoodyBlackWhiteFilter(BaseFilter):
    """Cinematic monochrome with vignette and optional grain."""

    name: str = "감성 흑백"
    description: str = (
        "강한 대비와 가장자리 비네팅, 약간의 필름 그레인을 더해 영화적인 느낌의 "
        "흑백 이미지를 만듭니다."
    )
    order: int = 111

    def render_controls(self) -> dict[str, Any]:
        contrast = st.sidebar.slider(
            "대비",
            min_value=0.5,
            max_value=2.5,
            value=1.3,
            step=0.05,
            key="moody_bw_contrast",
        )
        brightness = st.sidebar.slider(
            "밝기 보정",
            min_value=-60,
            max_value=60,
            value=-15,
            step=1,
            help="음수로 두면 더 어둡고 무거운 분위기가 됩니다.",
            key="moody_bw_brightness",
        )
        vignette = st.sidebar.slider(
            "비네팅 강도",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.05,
            help="가장자리를 얼마나 어둡게 할지 결정합니다.",
            key="moody_bw_vignette",
        )
        grain = st.sidebar.slider(
            "그레인 (필름 노이즈)",
            min_value=0,
            max_value=40,
            value=8,
            step=1,
            key="moody_bw_grain",
        )
        return {
            "contrast": float(contrast),
            "brightness": int(brightness),
            "vignette": float(vignette),
            "grain": int(grain),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        contrast: float = 1.3,
        brightness: int = -15,
        vignette: float = 0.7,
        grain: int = 8,
        **kwargs: Any,
    ) -> np.ndarray:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=float(brightness))

        if vignette > 0:
            mask = self._gaussian_vignette(gray.shape[:2], strength=vignette)
            gray = np.clip(gray.astype(np.float32) * mask, 0, 255).astype(np.uint8)

        if grain > 0:
            noise = np.random.normal(loc=0.0, scale=float(grain), size=gray.shape)
            gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _gaussian_vignette(shape: tuple[int, int], strength: float) -> np.ndarray:
        """Build a 2D Gaussian-shaped multiplicative vignette mask.

        Centre of the mask is ``1.0`` and corners drop towards
        ``1 - strength``. The result is intentionally clipped to ``[0, 1]``
        so very high strengths fully darken the corners instead of going
        negative.
        """
        h, w = shape
        ys, xs = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
        max_dist_sq = cx * cx + cy * cy
        falloff = (dist_sq / max_dist_sq).astype(np.float32)
        return np.clip(1.0 - strength * falloff, 0.0, 1.0)
