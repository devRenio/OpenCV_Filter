"""Glitch / VHS distortion effect filter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class GlitchFilter(BaseFilter):
    """Create a digital glitch / VHS corruption aesthetic.

    Effects applied:
    1. RGB channel horizontal shift (chromatic aberration on steroids)
    2. Random horizontal slice displacement
    3. Scanline noise bands
    4. Optional colour corruption
    """

    name: str = "글리치 효과"
    description: str = (
        "RGB 채널 분리, 가로줄 왜곡, 노이즈 밴드로 디지털 글리치/VHS 깨짐 "
        "효과를 만듭니다."
    )
    order: int = 500

    def render_controls(self) -> dict[str, Any]:
        intensity = st.sidebar.slider(
            "글리치 강도",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="전반적인 효과 강도를 조절합니다.",
            key="glitch_intensity",
        )
        channel_shift = st.sidebar.slider(
            "채널 분리 (RGB 시프트)",
            min_value=0,
            max_value=30,
            value=8,
            step=1,
            key="glitch_channel_shift",
        )
        slice_count = st.sidebar.slider(
            "슬라이스 개수",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            help="가로로 잘려서 어긋나는 영역 개수",
            key="glitch_slices",
        )
        noise_bands = st.sidebar.slider(
            "노이즈 밴드",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
            help="가로 노이즈 줄 개수",
            key="glitch_noise_bands",
        )
        color_corrupt = st.sidebar.checkbox(
            "색상 왜곡",
            value=True,
            help="일부 영역의 색상을 무작위로 변조합니다.",
            key="glitch_color_corrupt",
        )
        return {
            "intensity": float(intensity),
            "channel_shift": int(channel_shift),
            "slice_count": int(slice_count),
            "noise_bands": int(noise_bands),
            "color_corrupt": bool(color_corrupt),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        intensity: float = 0.5,
        channel_shift: int = 8,
        slice_count: int = 5,
        noise_bands: int = 3,
        color_corrupt: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]
        result = bgr.copy()

        # Scale parameters by intensity
        shift = int(channel_shift * intensity)
        slices = int(slice_count * intensity)
        bands = int(noise_bands * intensity)

        # 1. RGB channel shift (chromatic aberration)
        if shift > 0:
            result = self._channel_shift(result, shift)

        # 2. Horizontal slice displacement
        if slices > 0:
            result = self._slice_displacement(result, slices, intensity)

        # 3. Noise bands (horizontal lines of static)
        if bands > 0:
            result = self._add_noise_bands(result, bands, intensity)

        # 4. Color corruption in random blocks
        if color_corrupt and intensity > 0.2:
            result = self._color_corrupt(result, intensity)

        return result

    @staticmethod
    def _channel_shift(img: np.ndarray, shift: int) -> np.ndarray:
        """Shift R and B channels horizontally in opposite directions."""
        b, g, r = cv2.split(img)
        h, w = img.shape[:2]

        # Shift red channel right, blue channel left
        r_shifted = np.zeros_like(r)
        b_shifted = np.zeros_like(b)

        if shift < w:
            r_shifted[:, shift:] = r[:, :-shift]
            r_shifted[:, :shift] = r[:, -shift:]

            b_shifted[:, :-shift] = b[:, shift:]
            b_shifted[:, -shift:] = b[:, :shift]
        else:
            r_shifted = r
            b_shifted = b

        return cv2.merge([b_shifted, g, r_shifted])

    @staticmethod
    def _slice_displacement(img: np.ndarray, count: int, intensity: float) -> np.ndarray:
        """Randomly displace horizontal slices."""
        result = img.copy()
        h, w = img.shape[:2]
        max_offset = int(w * 0.15 * intensity)

        for _ in range(count):
            y_start = np.random.randint(0, h - 10)
            slice_height = np.random.randint(5, max(6, int(h * 0.08)))
            y_end = min(h, y_start + slice_height)
            offset = np.random.randint(-max_offset, max_offset + 1)

            if offset == 0:
                continue

            slice_data = img[y_start:y_end, :, :].copy()
            if offset > 0:
                result[y_start:y_end, offset:, :] = slice_data[:, :-offset, :]
                result[y_start:y_end, :offset, :] = slice_data[:, -offset:, :]
            else:
                offset = abs(offset)
                result[y_start:y_end, :-offset, :] = slice_data[:, offset:, :]
                result[y_start:y_end, -offset:, :] = slice_data[:, :offset, :]

        return result

    @staticmethod
    def _add_noise_bands(img: np.ndarray, count: int, intensity: float) -> np.ndarray:
        """Add horizontal bands of static noise."""
        result = img.copy()
        h, w = img.shape[:2]

        for _ in range(count):
            y_start = np.random.randint(0, h - 5)
            band_height = np.random.randint(2, max(3, int(h * 0.03)))
            y_end = min(h, y_start + band_height)

            noise = np.random.randint(0, 256, (y_end - y_start, w, 3), dtype=np.uint8)
            alpha = 0.3 + 0.5 * intensity
            result[y_start:y_end] = cv2.addWeighted(
                result[y_start:y_end], 1 - alpha, noise, alpha, 0
            )

        return result

    @staticmethod
    def _color_corrupt(img: np.ndarray, intensity: float) -> np.ndarray:
        """Randomly corrupt colors in small rectangular regions."""
        result = img.copy()
        h, w = img.shape[:2]
        num_blocks = int(3 * intensity)

        for _ in range(num_blocks):
            bx = np.random.randint(0, w - 20)
            by = np.random.randint(0, h - 10)
            bw = np.random.randint(20, min(100, w - bx))
            bh = np.random.randint(5, min(30, h - by))

            # Swap or boost a random channel
            channel = np.random.randint(0, 3)
            boost = np.random.choice([0.5, 1.5, 2.0])
            region = result[by:by + bh, bx:bx + bw].astype(np.float32)
            region[:, :, channel] = np.clip(region[:, :, channel] * boost, 0, 255)
            result[by:by + bh, bx:bx + bw] = region.astype(np.uint8)

        return result
