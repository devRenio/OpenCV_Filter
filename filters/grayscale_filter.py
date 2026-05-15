"""Grayscale filter implementation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.base_filter import BaseFilter
from core.registry import register


@register
class GrayscaleFilter(BaseFilter):
    """Convert a color image to grayscale.

    The output is intentionally returned as a 3-channel BGR image (with the
    same gray value replicated across the B, G and R channels) so the
    downstream rendering code can treat all filter outputs uniformly.
    """

    name: str = "Grayscale"
    description: str = (
        "Converts the image to grayscale using a luminance-preserving "
        "weighted average of the RGB channels."
    )

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the grayscale conversion to the input image.

        Args:
            image: Input image in BGR color order. May also accept a
                single-channel image, in which case it is returned as a
                3-channel BGR copy unchanged.
            **kwargs: Unused. Present to satisfy the :class:`BaseFilter`
                interface.

        Returns:
            A 3-channel BGR image whose channels all share the grayscale
            intensity value.
        """
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
