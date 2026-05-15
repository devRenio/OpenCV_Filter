"""Cartoon-style filter built from bilateral smoothing + adaptive edge mask."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from core.base_filter import BaseFilter
from core.registry import register


@register
class CartoonFilter(BaseFilter):
    """Reduce colour palette and overlay strong outlines for a comic look.

    Pipeline (the standard OpenCV cartoonisation recipe):

    1. Optionally downsample, since :func:`cv2.bilateralFilter` is the
       most expensive step.
    2. Run several iterations of bilateral filtering to flatten colour
       regions while preserving edges.
    3. From a grey, median-blurred copy of the image, extract a black
       outline mask via :func:`cv2.adaptiveThreshold`.
    4. Combine the smoothed colour image with the outline mask using
       :func:`cv2.bitwise_and`.
    """

    name: str = "만화 스타일"
    description: str = (
        "Bilateral 필터로 색을 단순화하고 적응형 임계값으로 추출한 검은 외곽선을 "
        "합성해 만화/일러스트 같은 느낌을 만듭니다."
    )
    order: int = 320

    # Cap the long edge of the working copy. bilateralFilter is O(N * sigma^2)
    # so processing a small image is essential for real-time webcam use.
    _MAX_PROCESS_SIDE: int = 480

    def render_controls(self) -> dict[str, Any]:
        smoothing = st.sidebar.slider(
            "평활화 단계",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            help="Bilateral 필터를 몇 번 반복할지 결정합니다. 높을수록 색이 더 평탄해집니다.",
            key="cartoon_smoothing",
        )
        sigma_color = st.sidebar.slider(
            "색 평활화 강도",
            min_value=10,
            max_value=150,
            value=75,
            step=5,
            key="cartoon_sigma_color",
        )
        edge_block = st.sidebar.slider(
            "엣지 블록 크기",
            min_value=3,
            max_value=15,
            value=9,
            step=2,
            help="외곽선 검출 윈도우 크기. 홀수만 허용됩니다.",
            key="cartoon_edge_block",
        )
        edge_strength = st.sidebar.slider(
            "엣지 강도",
            min_value=1,
            max_value=15,
            value=2,
            step=1,
            help="adaptiveThreshold의 C 값. 높을수록 외곽선이 가늘어집니다.",
            key="cartoon_edge_strength",
        )
        return {
            "smoothing": int(smoothing),
            "sigma_color": int(sigma_color),
            "edge_block": int(edge_block),
            "edge_strength": int(edge_strength),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        smoothing: int = 4,
        sigma_color: int = 75,
        edge_block: int = 9,
        edge_strength: int = 2,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        scale = max(1.0, max(h, w) / self._MAX_PROCESS_SIDE)
        if scale > 1.0:
            small = cv2.resize(bgr, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
        else:
            small = bgr

        smoothed = small
        for _ in range(max(1, smoothing)):
            smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=sigma_color, sigmaSpace=7)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        block = max(3, edge_block | 1)
        edges = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block,
            C=int(edge_strength),
        )
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon_small = cv2.bitwise_and(smoothed, edges_bgr)

        if scale > 1.0:
            return cv2.resize(cartoon_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cartoon_small
