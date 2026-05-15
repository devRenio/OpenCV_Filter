"""Pass-through "no-op" filter used as the default selection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.base_filter import BaseFilter
from core.registry import register


@register
class NoneFilter(BaseFilter):
    """Return the input image unchanged.

    Useful as a baseline for the side-by-side viewer and as the default
    selection in the sidebar when the user has not yet picked a real
    filter. ``order = 0`` guarantees it shows up first.
    """

    name: str = "원본 (필터 없음)"
    description: str = "필터를 적용하지 않고 원본 이미지를 그대로 출력합니다."
    order: int = 0

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Return the image unchanged, normalising to 3-channel BGR."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
