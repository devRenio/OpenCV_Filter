"""Streamlit entry point for the OpenCV image filter web app.

The app supports two input sources, each routed through the *same*
``BaseFilter.apply`` contract:

1. **Image Upload** - decode an uploaded file, run the filter once, and
   display the original / filtered result side by side.
2. **Webcam (Real-time)** - stream frames from the browser via
   ``streamlit-webrtc`` and run the same filter on every frame.

Channel-order convention used throughout this file:

* ``cv2``     -> BGR
* ``Pillow``  -> RGB
* ``streamlit.image`` -> RGB
* ``av.VideoFrame.to_ndarray(format="bgr24")`` -> BGR

To avoid color-swap bugs filters always receive BGR and return BGR. The
conversion to RGB happens only at the very last step before rendering.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Final

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

import filters  # noqa: F401  -- triggers filter auto-registration
from core.base_filter import BaseFilter
from core.registry import FilterRegistry

logger = logging.getLogger(__name__)

PAGE_TITLE: Final[str] = "OpenCV 이미지 필터 스튜디오"
ACCEPTED_TYPES: Final[list[str]] = ["png", "jpg", "jpeg", "bmp", "webp"]

# Public STUN server. Required for WebRTC to traverse NAT in remote
# deployments; localhost-only sessions usually work without it but we set
# it anyway for safety.
_RTC_CONFIG: Final[RTCConfiguration] = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_image_as_bgr(uploaded_file: io.BytesIO) -> np.ndarray:
    """Decode an uploaded file into a BGR NumPy array."""
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(pil_image, dtype=np.uint8)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (or 2D grayscale) image to RGB for ``st.image``."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# WebRTC video processor
# ---------------------------------------------------------------------------
class FilterVideoProcessor(VideoProcessorBase):
    """Apply a :class:`BaseFilter` to every webcam frame in real time.

    The instance attributes :attr:`filter_obj` and :attr:`parameters` are
    written from the Streamlit thread on every rerun (whenever a sidebar
    widget changes) and read from the WebRTC worker thread inside
    :meth:`recv`. Because we only ever swap whole references, no explicit
    locking is required for this read-mostly workload.
    """

    filter_obj: BaseFilter | None
    parameters: dict[str, Any]

    def __init__(self) -> None:
        self.filter_obj = None
        self.parameters = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single video frame.

        Args:
            frame: Incoming frame from the browser.

        Returns:
            Outgoing frame to be displayed in the browser.
        """
        bgr = frame.to_ndarray(format="bgr24")

        active_filter = self.filter_obj
        if active_filter is not None:
            try:
                bgr = active_filter.apply(bgr, **(self.parameters or {}))
            except Exception:  # noqa: BLE001 - filters are user-supplied plug-ins
                logger.exception("Filter %r failed on a webcam frame.", active_filter)

        # The filter contract guarantees BGR output. av re-encodes for us.
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple[str, BaseFilter | None, dict[str, Any]]:
    """Render the sidebar widgets.

    Returns:
        ``(mode, selected_filter, runtime_parameters)`` where ``mode`` is
        either ``"upload"`` or ``"webcam"``.
    """
    st.sidebar.title("설정")

    st.sidebar.subheader("1. 입력 소스")
    mode_options = {
        "이미지 업로드": "upload",
        "웹캠 (실시간)": "webcam",
    }
    mode_label = st.sidebar.radio(
        "모드",
        options=list(mode_options.keys()),
        index=0,
        key="input_mode",
    )
    mode = mode_options[mode_label]

    st.sidebar.subheader("2. 필터")
    registry = FilterRegistry()
    filter_names = registry.get_filter_names()
    if not filter_names:
        st.sidebar.warning(
            "등록된 필터가 없습니다. `filters/` 폴더에 `@register` 데코레이터가 "
            "적용된 모듈을 추가해 주세요."
        )
        return mode, None, {}

    selected_name = st.sidebar.selectbox(
        "필터 선택",
        options=filter_names,
        index=0,
        key="filter_select",
    )
    selected_filter = registry.get_filter_by_name(selected_name)
    st.sidebar.caption(selected_filter.description)

    st.sidebar.subheader("3. 필터 파라미터")
    parameters = selected_filter.render_controls()
    if not parameters:
        st.sidebar.caption("이 필터에는 조정 가능한 파라미터가 없습니다.")

    return mode, selected_filter, parameters


# ---------------------------------------------------------------------------
# Mode-specific renderers
# ---------------------------------------------------------------------------
def _render_upload_mode(
    selected_filter: BaseFilter,
    parameters: dict[str, Any],
) -> None:
    """Render the image-upload pipeline."""
    uploaded_file = st.file_uploader(
        "이미지 파일을 선택하세요",
        type=ACCEPTED_TYPES,
        help="지원 형식: " + ", ".join(ACCEPTED_TYPES),
    )
    if uploaded_file is None:
        st.info("시작하려면 이미지를 업로드해 주세요.")
        return

    try:
        original_bgr = _load_image_as_bgr(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - surface decoding errors to UI
        st.error(f"이미지를 읽을 수 없습니다: {exc}")
        return

    try:
        filtered_bgr = selected_filter.apply(original_bgr, **parameters)
    except Exception as exc:  # noqa: BLE001 - filters are user-supplied plug-ins
        st.error(f"'{selected_filter.name}' 필터 적용 중 오류가 발생했습니다: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**원본**")
        st.image(_bgr_to_rgb(original_bgr), use_container_width=True)
    with right:
        st.markdown(f"**필터 적용 - {selected_filter.name}**")
        st.image(_bgr_to_rgb(filtered_bgr), use_container_width=True)


def _render_webcam_mode(
    selected_filter: BaseFilter,
    parameters: dict[str, Any],
) -> None:
    """Render the real-time webcam pipeline."""
    st.markdown(
        "**START** 버튼을 눌러 카메라 접근을 허용하세요. 사이드바에서 필터를 "
        "변경하면 다음 렌더링부터 실시간 스트림에 즉시 반영됩니다."
    )

    ctx = webrtc_streamer(
        key="opencv-filter-webcam",
        video_processor_factory=FilterVideoProcessor,
        rtc_configuration=_RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Push the latest filter + parameters into the (background) processor.
    # This runs on every Streamlit rerun, i.e. whenever the user moves a
    # slider or flips a checkbox.
    if ctx.video_processor is not None:
        ctx.video_processor.filter_obj = selected_filter
        ctx.video_processor.parameters = parameters

    if ctx.state.playing:
        st.success(
            f"'{selected_filter.name}' 필터로 스트리밍 중입니다. 사이드바에서 "
            "파라미터를 조정하면 실시간으로 반영됩니다."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=None,
        layout="wide",
    )

    st.title(PAGE_TITLE)
    st.write(
        "이미지를 업로드하거나 웹캠 스트림에서 입력을 받은 뒤, 사이드바에서 "
        "필터를 선택하세요. `filters/` 폴더에 새 모듈을 추가하면 새로운 "
        "필터가 자동으로 등록됩니다."
    )

    mode, selected_filter, parameters = _render_sidebar()

    if selected_filter is None:
        st.warning("적용 가능한 필터가 없습니다.")
        return

    if mode == "webcam":
        _render_webcam_mode(selected_filter, parameters)
    else:
        _render_upload_mode(selected_filter, parameters)


if __name__ == "__main__":
    main()
