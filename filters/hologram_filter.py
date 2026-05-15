"""Sci-fi hologram projection effect filter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class HologramFilter(BaseFilter):
    """Create a sci-fi holographic projection aesthetic.

    Effects combined:
    1. Cyan/blue colour shift
    2. Horizontal scanlines
    3. Edge glow effect
    4. Slight transparency / ghosting
    5. Optional flicker simulation
    """

    name: str = "홀로그램"
    description: str = (
        "시안/청색 톤 + 스캔라인 + 엣지 글로우로 SF 영화의 홀로그램 통신 "
        "느낌을 만듭니다."
    )
    order: int = 530

    def render_controls(self) -> dict[str, Any]:
        tint_strength = st.sidebar.slider(
            "시안 색조 강도",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="hologram_tint",
        )
        scanline_strength = st.sidebar.slider(
            "스캔라인 강도",
            min_value=0.0,
            max_value=0.8,
            value=0.3,
            step=0.05,
            key="hologram_scanline",
        )
        scanline_gap = st.sidebar.slider(
            "스캔라인 간격",
            min_value=2,
            max_value=8,
            value=3,
            step=1,
            key="hologram_scanline_gap",
        )
        glow_strength = st.sidebar.slider(
            "엣지 글로우",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="hologram_glow",
        )
        ghost_strength = st.sidebar.slider(
            "고스팅 (잔상)",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="약간 시프트된 반투명 복제를 오버레이합니다.",
            key="hologram_ghost",
        )
        flicker = st.sidebar.checkbox(
            "플리커 효과",
            value=True,
            help="프레임마다 약간의 밝기 변동을 줍니다 (웹캠에서 효과적).",
            key="hologram_flicker",
        )
        return {
            "tint_strength": float(tint_strength),
            "scanline_strength": float(scanline_strength),
            "scanline_gap": int(scanline_gap),
            "glow_strength": float(glow_strength),
            "ghost_strength": float(ghost_strength),
            "flicker": bool(flicker),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        tint_strength: float = 0.7,
        scanline_strength: float = 0.3,
        scanline_gap: int = 3,
        glow_strength: float = 1.0,
        ghost_strength: float = 0.15,
        flicker: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        # 1. Apply cyan/blue tint
        result = self._apply_hologram_tint(bgr, tint_strength)

        # 2. Add edge glow
        if glow_strength > 0:
            result = self._add_edge_glow(result, bgr, glow_strength)

        # 3. Add ghosting (shifted transparent copy)
        if ghost_strength > 0:
            result = self._add_ghost(result, ghost_strength)

        # 4. Add scanlines
        if scanline_strength > 0:
            result = self._add_scanlines(result, scanline_strength, scanline_gap)

        # 5. Random flicker
        if flicker:
            result = self._add_flicker(result)

        return result

    @staticmethod
    def _apply_hologram_tint(img: np.ndarray, strength: float) -> np.ndarray:
        """Shift colours toward cyan/blue."""
        b, g, r = cv2.split(img)

        # Boost blue and green (cyan), reduce red
        b_new = np.clip(b.astype(np.float32) * (1.0 + 0.3 * strength), 0, 255)
        g_new = np.clip(g.astype(np.float32) * (1.0 + 0.2 * strength), 0, 255)
        r_new = np.clip(r.astype(np.float32) * (1.0 - 0.4 * strength), 0, 255)

        return cv2.merge([
            b_new.astype(np.uint8),
            g_new.astype(np.uint8),
            r_new.astype(np.uint8),
        ])

    @staticmethod
    def _add_edge_glow(
        tinted: np.ndarray, original: np.ndarray, strength: float
    ) -> np.ndarray:
        """Add glowing edges extracted via Laplacian."""
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.abs(edges)
        edges = np.clip(edges / edges.max() * 255 if edges.max() > 0 else edges, 0, 255)
        edges = edges.astype(np.uint8)

        # Blur edges for glow effect
        glow = cv2.GaussianBlur(edges, (0, 0), sigmaX=3)

        # Make glow cyan-ish
        glow_bgr = np.zeros_like(tinted)
        glow_bgr[:, :, 0] = glow  # Blue
        glow_bgr[:, :, 1] = (glow * 0.8).astype(np.uint8)  # Green

        # Additive blend
        result = cv2.addWeighted(tinted, 1.0, glow_bgr, strength * 0.5, 0)
        return result

    @staticmethod
    def _add_ghost(img: np.ndarray, strength: float) -> np.ndarray:
        """Add a slightly shifted transparent copy for ghosting effect."""
        h, w = img.shape[:2]
        shift_x = int(w * 0.02)
        shift_y = int(h * 0.01)

        ghost = np.zeros_like(img)
        if shift_x < w and shift_y < h:
            ghost[shift_y:, shift_x:, :] = img[:-shift_y, :-shift_x, :]

        return cv2.addWeighted(img, 1.0, ghost, strength, 0)

    @staticmethod
    def _add_scanlines(
        img: np.ndarray, strength: float, gap: int
    ) -> np.ndarray:
        """Overlay horizontal scanlines."""
        result = img.copy().astype(np.float32)
        h = img.shape[0]
        gap = max(2, gap)

        # Darken every nth line
        for y in range(0, h, gap):
            result[y, :, :] *= (1.0 - strength)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _add_flicker(img: np.ndarray) -> np.ndarray:
        """Add random brightness variation to simulate unstable projection."""
        variation = np.random.uniform(0.92, 1.08)
        result = img.astype(np.float32) * variation
        return np.clip(result, 0, 255).astype(np.uint8)
