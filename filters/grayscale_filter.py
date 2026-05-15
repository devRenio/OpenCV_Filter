"""Grayscale filter implementation with tunable parameters."""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

# Mapping from UI label to internal method id. Defining it once at module
# level keeps both the widget and the apply() implementation in sync.
_METHODS: Final[dict[str, str]] = {
    "광도 가중 평균 (BT.601)": "bt601",
    "HDTV (BT.709)": "bt709",
    "산술 평균": "average",
    "명도 (HSL)": "lightness",
    "Red 채널만": "red",
    "Green 채널만": "green",
    "Blue 채널만": "blue",
}


@register
class GrayscaleFilter(BaseFilter):
    """Convert a color image to grayscale with adjustable parameters.

    Parameters exposed in :meth:`render_controls`:

    * ``method``     -- conversion formula (luminosity / BT.709 / average /
                        lightness / single-channel extraction).
    * ``brightness`` -- additive offset applied to every pixel.
    * ``contrast``   -- multiplicative gain applied around 0.
    * ``invert``     -- whether to invert the resulting intensity.

    The output is always a 3-channel BGR image so the rest of the
    pipeline can treat all filter outputs uniformly.
    """

    name: str = "흑백"
    description: str = (
        "선택한 변환 공식을 이용해 이미지를 흑백으로 변환하고, 밝기·대비·반전 "
        "값을 자유롭게 조정할 수 있습니다."
    )
    order: int = 110

    def render_controls(self) -> dict[str, Any]:
        """Render filter-specific Streamlit widgets.

        Returns:
            Keyword arguments that will be forwarded to :meth:`apply`.
        """
        method_label = st.sidebar.selectbox(
            "변환 공식",
            options=list(_METHODS.keys()),
            index=0,
            help=(
                "BT.601은 OpenCV 기본값이며, BT.709는 HDTV 표준에 맞춰진 "
                "공식입니다. 산술 평균과 명도(HSL)는 사람의 시각 특성을 "
                "반영하지 않습니다. 단일 채널 모드는 BGR 중 한 채널만 그대로 "
                "사용합니다."
            ),
            key="grayscale_method",
        )
        brightness = st.sidebar.slider(
            "밝기",
            min_value=-100,
            max_value=100,
            value=0,
            step=1,
            key="grayscale_brightness",
        )
        contrast = st.sidebar.slider(
            "대비",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="grayscale_contrast",
        )
        invert = st.sidebar.checkbox(
            "반전 (Invert)",
            value=False,
            key="grayscale_invert",
        )
        return {
            "method": _METHODS[method_label],
            "brightness": brightness,
            "contrast": contrast,
            "invert": invert,
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        method: str = "bt601",
        brightness: int = 0,
        contrast: float = 1.0,
        invert: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply the grayscale conversion to the input image.

        Args:
            image: Input image in BGR color order. A 2D grayscale array is
                also accepted and returned as a 3-channel BGR image after
                applying the post-processing steps.
            method: Identifier of the conversion formula. See ``_METHODS``.
            brightness: Additive offset (``-100..100``) applied per pixel.
            contrast: Multiplicative gain (``0.1..3.0``) applied per pixel.
            invert: When True, invert the intensity (255 - pixel).
            **kwargs: Reserved for forward compatibility. Ignored.

        Returns:
            A 3-channel BGR image with the requested transformations
            applied.
        """
        gray = self._to_gray(image, method)

        if contrast != 1.0 or brightness != 0:
            # convertScaleAbs handles overflow / underflow and casts to uint8
            gray = cv2.convertScaleAbs(gray, alpha=float(contrast), beta=float(brightness))

        if invert:
            gray = cv2.bitwise_not(gray)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _to_gray(image: np.ndarray, method: str) -> np.ndarray:
        """Reduce ``image`` to a single-channel grayscale array.

        Args:
            image: BGR image, or already-grayscale 2D array.
            method: Conversion identifier from ``_METHODS``.

        Returns:
            A 2D ``uint8`` array.
        """
        if image.ndim == 2:
            return image

        b, g, r = cv2.split(image)

        if method == "bt601":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == "bt709":
            gray = 0.0722 * b.astype(np.float32) + 0.7152 * g.astype(np.float32) + 0.2126 * r.astype(np.float32)
            return np.clip(gray, 0, 255).astype(np.uint8)
        if method == "average":
            stack = np.stack([b, g, r], axis=-1).astype(np.uint16)
            return (stack.sum(axis=-1) // 3).astype(np.uint8)
        if method == "lightness":
            stack = np.stack([b, g, r], axis=-1)
            hi = stack.max(axis=-1).astype(np.uint16)
            lo = stack.min(axis=-1).astype(np.uint16)
            return ((hi + lo) // 2).astype(np.uint8)
        if method == "red":
            return r
        if method == "green":
            return g
        if method == "blue":
            return b

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
