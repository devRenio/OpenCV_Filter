"""CCTV / surveillance camera style filter."""

from __future__ import annotations

import datetime as _dt
from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class CCTVFilter(BaseFilter):
    """Low-resolution monochrome look with noise, scanlines and timestamp.

    The pipeline mimics the artefacts of cheap analogue surveillance
    cameras: low resolution, sensor noise, interlaced scanlines and a
    burned-in date/time string in a corner.
    """

    name: str = "CCTV 스타일"
    description: str = (
        "저해상도 흑백, 센서 노이즈, 스캔라인, 날짜·시간 오버레이로 감시 카메라 "
        "느낌을 만듭니다."
    )
    order: int = 160

    def render_controls(self) -> dict[str, Any]:
        downscale = st.sidebar.slider(
            "해상도 다운스케일",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help="값이 클수록 더 거칠고 저화질로 보입니다.",
            key="cctv_downscale",
        )
        noise = st.sidebar.slider(
            "노이즈 강도",
            min_value=0,
            max_value=60,
            value=15,
            step=1,
            key="cctv_noise",
        )
        scanline = st.sidebar.slider(
            "스캔라인 강도",
            min_value=0.0,
            max_value=0.6,
            value=0.15,
            step=0.05,
            key="cctv_scanline",
        )
        show_timestamp = st.sidebar.checkbox(
            "날짜·시간 오버레이",
            value=True,
            key="cctv_timestamp",
        )
        show_rec = st.sidebar.checkbox(
            "녹화중(REC) 인디케이터",
            value=True,
            key="cctv_rec",
        )
        return {
            "downscale": int(downscale),
            "noise": int(noise),
            "scanline": float(scanline),
            "show_timestamp": bool(show_timestamp),
            "show_rec": bool(show_rec),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        downscale: int = 3,
        noise: int = 15,
        scanline: float = 0.15,
        show_timestamp: bool = True,
        show_rec: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=-5.0)

        # Downsample then upsample to crush detail like an old analogue feed.
        d = max(1, int(downscale))
        if d > 1:
            small = cv2.resize(gray, (max(1, w // d), max(1, h // d)), interpolation=cv2.INTER_LINEAR)
            gray = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        if noise > 0:
            n = np.random.normal(loc=0.0, scale=float(noise), size=gray.shape)
            gray = np.clip(gray.astype(np.float32) + n, 0, 255).astype(np.uint8)

        if scanline > 0:
            mask = np.ones_like(gray, dtype=np.float32)
            mask[::2, :] = 1.0 - scanline
            gray = np.clip(gray.astype(np.float32) * mask, 0, 255).astype(np.uint8)

        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if show_timestamp:
            self._draw_timestamp(out)
        if show_rec:
            self._draw_rec_indicator(out)
        return out

    @staticmethod
    def _adaptive_scale(canvas: np.ndarray) -> float:
        """Pick a font scale proportional to the image's short side.

        At ~500px the scale is 1.0; the result is clamped to a
        readable range so very small thumbnails do not get unreadable
        sub-pixel text and 4K frames do not get giant overlays.
        """
        h, w = canvas.shape[:2]
        return float(max(0.8, min(3.0, min(h, w) / 500.0)))

    @classmethod
    def _draw_timestamp(cls, canvas: np.ndarray) -> None:
        """Burn the current local date/time into the bottom-left corner."""
        text = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h, _ = canvas.shape[:2]
        scale = cls._adaptive_scale(canvas) * 1.1
        thickness = max(2, int(round(scale * 1.8)))
        margin_x = int(round(scale * 18))
        margin_y = int(round(scale * 22))
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (margin_x, h - margin_y)
        # Black outline + white text -> readable on any background.
        cv2.putText(canvas, text, origin, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    @classmethod
    def _draw_rec_indicator(cls, canvas: np.ndarray) -> None:
        """Draw a red dot + 'REC' label in the top-right corner."""
        h, w = canvas.shape[:2]
        scale = cls._adaptive_scale(canvas) * 1.3
        thickness = max(2, int(round(scale * 2.2)))
        radius = max(8, int(round(scale * 10)))
        margin_x = int(round(scale * 18))
        margin_y = int(round(scale * 18))

        # Approximate the width of "REC" so the circle sits to its left.
        (text_w, text_h), baseline = cv2.getTextSize(
            "REC", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        text_origin = (w - margin_x - text_w, margin_y + text_h)
        circle_center = (text_origin[0] - radius - int(scale * 6), margin_y + text_h - radius // 2)

        cv2.circle(canvas, circle_center, radius, (0, 0, 220), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas, "REC", text_origin,
            cv2.FONT_HERSHEY_SIMPLEX, scale,
            (0, 0, 0), thickness + 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "REC", text_origin,
            cv2.FONT_HERSHEY_SIMPLEX, scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )
