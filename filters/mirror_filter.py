"""Mirror and kaleidoscope effect filter."""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

_MIRROR_MODES: Final[dict[str, str]] = {
    "좌우 대칭 (왼쪽 기준)": "left_to_right",
    "좌우 대칭 (오른쪽 기준)": "right_to_left",
    "상하 대칭 (위쪽 기준)": "top_to_bottom",
    "상하 대칭 (아래쪽 기준)": "bottom_to_top",
    "4분할 대칭": "quad",
    "만화경 (6각형)": "kaleidoscope",
}


@register
class MirrorFilter(BaseFilter):
    """Apply various mirror and kaleidoscope symmetry effects.

    Modes include simple horizontal/vertical mirroring, 4-way quad
    symmetry, and a hexagonal kaleidoscope pattern.
    """

    name: str = "거울/만화경"
    description: str = (
        "좌우, 상하, 4분할 대칭 또는 만화경 패턴으로 이미지를 변환합니다. "
        "웹캠에서 특히 재미있습니다."
    )
    order: int = 520

    def render_controls(self) -> dict[str, Any]:
        mode_label = st.sidebar.selectbox(
            "대칭 모드",
            options=list(_MIRROR_MODES.keys()),
            index=0,
            key="mirror_mode",
        )
        mode = _MIRROR_MODES[mode_label]

        offset = 0
        if mode in ("left_to_right", "right_to_left"):
            offset = st.sidebar.slider(
                "대칭축 위치 (%)",
                min_value=20,
                max_value=80,
                value=50,
                step=1,
                help="좌우 대칭의 중심선 위치",
                key="mirror_offset_h",
            )
        elif mode in ("top_to_bottom", "bottom_to_top"):
            offset = st.sidebar.slider(
                "대칭축 위치 (%)",
                min_value=20,
                max_value=80,
                value=50,
                step=1,
                help="상하 대칭의 중심선 위치",
                key="mirror_offset_v",
            )

        segments = 6
        if mode == "kaleidoscope":
            segments = st.sidebar.slider(
                "만화경 분할 수",
                min_value=4,
                max_value=12,
                value=6,
                step=2,
                key="mirror_segments",
            )

        return {
            "mode": mode,
            "offset": int(offset),
            "segments": int(segments),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        mode: str = "left_to_right",
        offset: int = 50,
        segments: int = 6,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if mode == "left_to_right":
            return self._mirror_horizontal(bgr, offset, left_source=True)
        elif mode == "right_to_left":
            return self._mirror_horizontal(bgr, offset, left_source=False)
        elif mode == "top_to_bottom":
            return self._mirror_vertical(bgr, offset, top_source=True)
        elif mode == "bottom_to_top":
            return self._mirror_vertical(bgr, offset, top_source=False)
        elif mode == "quad":
            return self._mirror_quad(bgr)
        elif mode == "kaleidoscope":
            return self._kaleidoscope(bgr, segments)

        return bgr

    @staticmethod
    def _mirror_horizontal(
        img: np.ndarray, offset_pct: int, left_source: bool
    ) -> np.ndarray:
        """Mirror left half to right or vice versa."""
        h, w = img.shape[:2]
        mid = int(w * offset_pct / 100.0)
        result = img.copy()

        if left_source:
            # Copy left side, flip it to right side
            left = img[:, :mid, :]
            flipped = cv2.flip(left, 1)
            # Resize flipped to fit right side
            right_width = w - mid
            if flipped.shape[1] != right_width:
                flipped = cv2.resize(flipped, (right_width, h))
            result[:, mid:, :] = flipped
        else:
            # Copy right side, flip it to left side
            right = img[:, mid:, :]
            flipped = cv2.flip(right, 1)
            left_width = mid
            if flipped.shape[1] != left_width:
                flipped = cv2.resize(flipped, (left_width, h))
            result[:, :mid, :] = flipped

        return result

    @staticmethod
    def _mirror_vertical(
        img: np.ndarray, offset_pct: int, top_source: bool
    ) -> np.ndarray:
        """Mirror top half to bottom or vice versa."""
        h, w = img.shape[:2]
        mid = int(h * offset_pct / 100.0)
        result = img.copy()

        if top_source:
            top = img[:mid, :, :]
            flipped = cv2.flip(top, 0)
            bottom_height = h - mid
            if flipped.shape[0] != bottom_height:
                flipped = cv2.resize(flipped, (w, bottom_height))
            result[mid:, :, :] = flipped
        else:
            bottom = img[mid:, :, :]
            flipped = cv2.flip(bottom, 0)
            top_height = mid
            if flipped.shape[0] != top_height:
                flipped = cv2.resize(flipped, (w, top_height))
            result[:mid, :, :] = flipped

        return result

    @staticmethod
    def _mirror_quad(img: np.ndarray) -> np.ndarray:
        """Create 4-way symmetry from top-left quadrant."""
        h, w = img.shape[:2]
        mid_y, mid_x = h // 2, w // 2

        # Take top-left quadrant as source
        quad = img[:mid_y, :mid_x, :]

        result = np.zeros_like(img)

        # Top-left: original
        result[:mid_y, :mid_x, :] = quad
        # Top-right: horizontal flip
        result[:mid_y, mid_x:, :] = cv2.flip(
            cv2.resize(quad, (w - mid_x, mid_y)), 1
        )
        # Bottom-left: vertical flip
        result[mid_y:, :mid_x, :] = cv2.flip(
            cv2.resize(quad, (mid_x, h - mid_y)), 0
        )
        # Bottom-right: both flips
        result[mid_y:, mid_x:, :] = cv2.flip(
            cv2.resize(quad, (w - mid_x, h - mid_y)), -1
        )

        return result

    @staticmethod
    def _kaleidoscope(img: np.ndarray, segments: int) -> np.ndarray:
        """Create a kaleidoscope effect by rotating and mirroring a wedge.

        This uses polar coordinate remapping to create the effect.
        """
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        max_r = int(np.sqrt(cx * cx + cy * cy))

        # Create coordinate grids
        y_grid, x_grid = np.ogrid[:h, :w]
        x_centered = x_grid - cx
        y_centered = y_grid - cy

        # Convert to polar coordinates
        r = np.sqrt(x_centered ** 2 + y_centered ** 2).astype(np.float32)
        theta = np.arctan2(y_centered, x_centered).astype(np.float32)

        # Normalize theta to [0, 2*pi)
        theta = np.mod(theta, 2 * np.pi)

        # Calculate segment angle
        segment_angle = 2 * np.pi / segments

        # Fold theta into first segment, with mirroring
        segment_idx = (theta / segment_angle).astype(int)
        local_theta = theta - segment_idx * segment_angle

        # Mirror odd segments
        mirror_mask = (segment_idx % 2) == 1
        local_theta[mirror_mask] = segment_angle - local_theta[mirror_mask]

        # Convert back to Cartesian (relative to source image center)
        src_x = (cx + r * np.cos(local_theta)).astype(np.float32)
        src_y = (cy + r * np.sin(local_theta)).astype(np.float32)

        # Clamp coordinates
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        # Remap
        result = cv2.remap(
            img, src_x, src_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return result
