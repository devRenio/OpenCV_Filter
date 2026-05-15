"""Abstract base class definition for image filters.

This module defines the contract that every concrete filter implementation
must follow. By relying on a single abstract interface, the application can
treat all filters polymorphically and integrate new ones without changing
existing UI or pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseFilter(ABC):
    """Abstract base class for all image filters.

    Every concrete filter must inherit from this class and implement the
    :meth:`apply` method. Subclasses are expected to declare the class-level
    attributes :attr:`name` and :attr:`description` which are consumed by the
    UI layer (Streamlit) for display purposes.

    Attributes:
        name: Human-readable display name shown in the UI.
        description: Short explanation of what the filter does.
        order: Sort key used by the registry to control the position of
            the filter inside the UI selectbox. Lower values appear
            first; ties are broken alphabetically by ``name``. The
            default of ``100`` places a filter in the middle, leaving
            room for "header" filters (e.g. a passthrough) to use
            ``0`` and category groupings to use sparse buckets.
    """

    name: str = "기본 필터"
    description: str = "추상 베이스 필터입니다. 서브클래스에서 구현해 주세요."
    order: int = 100

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the filter to the given image.

        Args:
            image: Input image as a NumPy array in **BGR** color order
                (OpenCV convention). Shape is ``(H, W, 3)`` for color images
                or ``(H, W)`` for grayscale images.
            **kwargs: Optional filter-specific parameters (e.g., kernel size,
                threshold values) that may be supplied by the UI.

        Returns:
            The filtered image as a NumPy array in **BGR** color order.
            Concrete filters that internally produce a different layout
            (e.g., a single-channel grayscale image) are responsible for
            converting the result back to a 3-channel BGR image so the
            downstream pipeline can treat all outputs uniformly.
        """

    def render_controls(self) -> dict[str, Any]:
        """Render Streamlit widgets for filter-specific parameters.

        Subclasses can override this hook to expose tunable knobs in the
        sidebar. The default implementation returns an empty mapping,
        meaning the filter has no user-controllable parameters.

        Returns:
            A dictionary of keyword arguments forwarded to :meth:`apply`.
        """
        return {}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self.__class__.__name__} name={self.name!r}>"
