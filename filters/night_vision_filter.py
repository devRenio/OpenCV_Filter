"""Night-vision goggles style filter (green channel + noise + vignette)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class NightVisionFilter(BaseFilter):
    """Map the image to a green channel with noise, scanlines and vignette.

    The classic night-vision goggle look is approximated by:

    1. Converting to grayscale and pushing the contrast / brightness to
       simulate amplified low-light footage.
    2. Mapping the result onto only the green BGR channel so the image
       glows green.
    3. Adding Gaussian noise (sensor amplification noise) and optional
       horizontal scanlines (CRT-style raster).
    4. A soft round vignette to mimic the goggle aperture.
    """

    name: str = "야간투시경"
    description: str = (
        "녹색 단색 매핑과 센서 노이즈, 스캔라인, 원형 비네팅으로 야간투시경 "
        "느낌을 구현합니다."
    )
    order: int = 150

    def render_controls(self) -> dict[str, Any]:
        intensity = st.sidebar.slider(
            "광량 증폭",
            min_value=0.5,
            max_value=3.0,
            value=1.6,
            step=0.05,
            key="night_vision_intensity",
        )
        noise = st.sidebar.slider(
            "노이즈 강도",
            min_value=0,
            max_value=60,
            value=20,
            step=1,
            key="night_vision_noise",
        )
        scanline = st.sidebar.slider(
            "스캔라인 강도",
            min_value=0.0,
            max_value=0.6,
            value=0.2,
            step=0.05,
            help="0이면 스캔라인 없음. 클수록 짝수 줄이 더 어두워집니다.",
            key="night_vision_scanline",
        )
        vignette = st.sidebar.slider(
            "비네팅 강도",
            min_value=0.0,
            max_value=1.5,
            value=0.9,
            step=0.05,
            key="night_vision_vignette",
        )
        return {
            "intensity": float(intensity),
            "noise": int(noise),
            "scanline": float(scanline),
            "vignette": float(vignette),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        intensity: float = 1.6,
        noise: int = 20,
        scanline: float = 0.2,
        vignette: float = 0.9,
        **kwargs: Any,
    ) -> np.ndarray:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=intensity, beta=10.0)

        if noise > 0:
            n = np.random.normal(loc=0.0, scale=float(noise), size=gray.shape)
            gray = np.clip(gray.astype(np.float32) + n, 0, 255).astype(np.uint8)

        if scanline > 0:
            mask = np.ones_like(gray, dtype=np.float32)
            mask[::2, :] = 1.0 - scanline
            gray = np.clip(gray.astype(np.float32) * mask, 0, 255).astype(np.uint8)

        if vignette > 0:
            v = self._vignette_mask(gray.shape, strength=vignette)
            gray = np.clip(gray.astype(np.float32) * v, 0, 255).astype(np.uint8)

        zeros = np.zeros_like(gray)
        # OpenCV merge order is B, G, R -> push intensity into G only.
        return cv2.merge([zeros, gray, zeros])

    @staticmethod
    def _vignette_mask(shape: tuple[int, int], strength: float) -> np.ndarray:
        h, w = shape
        ys, xs = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        max_d2 = cx * cx + cy * cy
        return np.clip(1.0 - strength * (d2 / max_d2).astype(np.float32), 0.0, 1.0)
