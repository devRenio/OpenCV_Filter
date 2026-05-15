"""Color inversion (negative) filter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.base_filter import BaseFilter
from core.registry import register


@register
class InvertFilter(BaseFilter):
    """Invert each pixel value -> ``255 - x`` (negative effect)."""

    name: str = "색상 반전"
    description: str = "각 채널의 픽셀 값을 (255 - x)로 반전해 네거티브 효과를 만듭니다."
    order: int = 120

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Bitwise-not of the image, returning a 3-channel BGR result."""
        inverted = cv2.bitwise_not(image)
        if inverted.ndim == 2:
            return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        return inverted
