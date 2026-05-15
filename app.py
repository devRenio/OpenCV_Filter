"""Streamlit entry point for the OpenCV image filter web app.

The app loads an image uploaded by the user, asks the registry which
filters are available, applies the selected filter and renders the
original and filtered images side by side for comparison.

Channel-order convention used throughout this file:

* ``cv2``     -> BGR
* ``Pillow``  -> RGB
* ``numpy``   -> Whatever was last assigned (we are explicit about it)
* ``streamlit.image`` -> RGB

To avoid color-swap bugs we *always* feed BGR to filters and convert to
RGB only at the very end, just before rendering.
"""

from __future__ import annotations

import io
from typing import Final

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import filters  # noqa: F401  -- triggers filter auto-registration
from core.base_filter import BaseFilter
from core.registry import FilterRegistry

PAGE_TITLE: Final[str] = "OpenCV Image Filter Studio"
ACCEPTED_TYPES: Final[list[str]] = ["png", "jpg", "jpeg", "bmp", "webp"]


def _load_image_as_bgr(uploaded_file: io.BytesIO) -> np.ndarray:
    """Decode an uploaded file into a BGR NumPy array.

    Pillow is used for decoding because it copes with a wider range of
    formats than :func:`cv2.imdecode` out of the box. The image is then
    converted from RGB to BGR so that the rest of the pipeline can stick
    to OpenCV's native channel order.

    Args:
        uploaded_file: File-like object returned by ``st.file_uploader``.

    Returns:
        A 3-channel BGR image as a ``uint8`` NumPy array.
    """
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(pil_image, dtype=np.uint8)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for Streamlit rendering.

    Args:
        image: Either a 2D grayscale image or a 3-channel BGR image.

    Returns:
        A 3-channel RGB image suitable for ``st.image``.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _render_sidebar() -> tuple[io.BytesIO | None, BaseFilter | None]:
    """Render the sidebar widgets and return the user's selections.

    Returns:
        A tuple of ``(uploaded_file, selected_filter)``. Either element
        may be ``None`` if the user has not made the corresponding
        selection yet.
    """
    st.sidebar.title("Controls")

    st.sidebar.subheader("1. Upload an image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=ACCEPTED_TYPES,
        help="Supported formats: " + ", ".join(ACCEPTED_TYPES),
    )

    st.sidebar.subheader("2. Pick a filter")
    registry = FilterRegistry()
    filter_names = registry.get_filter_names()

    if not filter_names:
        st.sidebar.warning(
            "No filters are registered. Add a module under `filters/` "
            "decorated with `@register` to get started."
        )
        return uploaded_file, None

    selected_name = st.sidebar.selectbox(
        "Filter",
        options=filter_names,
        index=0,
    )
    selected_filter = registry.get_filter_by_name(selected_name)

    st.sidebar.caption(selected_filter.description)

    st.sidebar.subheader("3. Filter parameters")
    parameters = selected_filter.render_controls()
    if not parameters:
        st.sidebar.caption("This filter has no adjustable parameters.")

    selected_filter.runtime_parameters = parameters  # type: ignore[attr-defined]
    return uploaded_file, selected_filter


def _render_comparison(original_bgr: np.ndarray, filtered_bgr: np.ndarray) -> None:
    """Render the original and filtered images side by side.

    Args:
        original_bgr: Original image in BGR color order.
        filtered_bgr: Filtered image in BGR color order.
    """
    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.image(_bgr_to_rgb(original_bgr), use_container_width=True)
    with right:
        st.markdown("**Filtered**")
        st.image(_bgr_to_rgb(filtered_bgr), use_container_width=True)


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=None,
        layout="wide",
    )

    st.title(PAGE_TITLE)
    st.write(
        "Upload an image, pick a filter from the sidebar and compare the "
        "result with the original side by side. New filters can be added "
        "by dropping a new module into the `filters/` folder."
    )

    uploaded_file, selected_filter = _render_sidebar()

    if uploaded_file is None:
        st.info("Upload an image from the sidebar to begin.")
        return

    if selected_filter is None:
        st.warning("No filter is available to apply.")
        return

    try:
        original_bgr = _load_image_as_bgr(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - surface decoding errors to UI
        st.error(f"Could not read the uploaded image: {exc}")
        return

    runtime_parameters: dict = getattr(selected_filter, "runtime_parameters", {}) or {}

    try:
        filtered_bgr = selected_filter.apply(original_bgr, **runtime_parameters)
    except Exception as exc:  # noqa: BLE001 - filters are user-supplied plug-ins
        st.error(f"Filter '{selected_filter.name}' failed: {exc}")
        return

    _render_comparison(original_bgr, filtered_bgr)


if __name__ == "__main__":
    main()
