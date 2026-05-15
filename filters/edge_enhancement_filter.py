"""Edge enhancement filter (keeps the original colours, just amplifies edges)."""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register

_EDGE_OPERATORS: Final[tuple[str, ...]] = ("Laplacian", "Sobel", "Scharr")


@register
class EdgeEnhancementFilter(BaseFilter):
    """Amplify image edges while preserving the original colour content.

    Different from :class:`CannyEdgeFilter` (which *replaces* the image
    with the edge map), this filter computes a signed edge response and
    *adds* it back to the original BGR image:

        result = original + strength * edge_response

    The edge operator is selectable so the user can pick the response
    that best matches their image (Laplacian for fine textures, Sobel
    for stronger gradients, Scharr for the smoothest result).
    """

    name: str = "엣지 강조"
    description: str = (
        "원본 색상은 유지하면서 Laplacian/Sobel/Scharr로 추출한 엣지를 더해 "
        "윤곽이 또렷한 이미지로 만듭니다."
    )
    order: int = 230

    def render_controls(self) -> dict[str, Any]:
        operator = st.sidebar.selectbox(
            "엣지 검출 방식",
            options=_EDGE_OPERATORS,
            index=0,
            key="edge_enhance_operator",
        )
        strength = st.sidebar.slider(
            "강조 강도",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="0이면 원본, 클수록 엣지가 더 두드러집니다.",
            key="edge_enhance_strength",
        )
        ksize = st.sidebar.slider(
            "커널 크기",
            min_value=1,
            max_value=7,
            value=3,
            step=2,
            help="홀수만 허용됩니다. Scharr는 항상 3x3을 사용합니다.",
            key="edge_enhance_kernel",
        )
        return {
            "operator": str(operator),
            "strength": float(strength),
            "ksize": int(ksize),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        operator: str = "Laplacian",
        strength: float = 1.0,
        ksize: int = 3,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if strength <= 0:
            return bgr

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        k = max(1, ksize | 1)

        if operator == "Sobel":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
            edge = cv2.magnitude(gx, gy)
        elif operator == "Scharr":
            gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            edge = cv2.magnitude(gx, gy)
        else:  # Laplacian (default)
            edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=k)
            edge = np.abs(edge)

        # Normalise to [0, 255] so 'strength' has a comparable meaning
        # across operators.
        edge_max = float(edge.max())
        if edge_max > 0:
            edge = edge * (255.0 / edge_max)

        edge_bgr = cv2.cvtColor(edge.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        boosted = cv2.addWeighted(bgr, 1.0, edge_bgr, float(strength), 0.0)
        return boosted
