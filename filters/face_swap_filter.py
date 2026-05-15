"""Face swap filter using YuNet landmarks and seamless cloning.

Two modes are supported:

1. **Swap within image** - If two or more faces are detected in the
   current frame, swap their faces with each other (round-robin).
2. **Swap with reference** - Upload a reference image in the sidebar;
   every detected face in the main image/webcam is replaced with the
   reference face.

Both modes use YuNet's 5-point landmarks (eyes, nose, mouth corners)
to compute an affine transform that aligns the source face onto the
target, then :func:`cv2.seamlessClone` blends it naturally.
"""

from __future__ import annotations

import io
import logging
from typing import Any, ClassVar

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.base_filter import BaseFilter
from core.registry import register

# Reuse the YuNet detector infrastructure from face_mosaic_filter.
# We import the helper and detector class directly.
from filters.face_mosaic_filter import _YuNetDetector, _ensure_model

logger = logging.getLogger(__name__)


def _get_yunet_landmarks(
    bgr: np.ndarray,
    confidence_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect faces and extract 5-point landmarks via YuNet.

    Returns:
        A list of dicts, each containing:
        - ``box``: (x, y, w, h)
        - ``landmarks``: np.ndarray of shape (5, 2) — left_eye, right_eye,
          nose, mouth_left, mouth_right in pixel coordinates.
        - ``confidence``: float
    """
    try:
        detector = _YuNetDetector.get()
    except Exception as exc:
        logger.warning("YuNet 초기화 실패: %s", exc)
        return []

    h, w = bgr.shape[:2]
    detector._detector.setInputSize((w, h))
    detector._detector.setScoreThreshold(float(confidence_threshold))

    _, faces = detector._detector.detect(bgr)
    if faces is None:
        return []

    results: list[dict[str, Any]] = []
    for face in faces:
        x, y, fw, fh = face[0:4]
        # Landmarks: indices 4-13 are (eye_l_x, eye_l_y, eye_r_x, eye_r_y,
        # nose_x, nose_y, mouth_l_x, mouth_l_y, mouth_r_x, mouth_r_y)
        lm = face[4:14].reshape(5, 2)
        conf = float(face[14]) if len(face) > 14 else float(face[-1])
        results.append({
            "box": (int(max(0, x)), int(max(0, y)), int(fw), int(fh)),
            "landmarks": lm.astype(np.float32),
            "confidence": conf,
        })
    return results


def _compute_affine_from_landmarks(
    src_lm: np.ndarray,
    dst_lm: np.ndarray,
) -> np.ndarray:
    """Compute 2×3 affine matrix mapping ``src_lm`` onto ``dst_lm``.

    Uses the 3 most stable points: left_eye (0), right_eye (1), nose (2).
    """
    src_pts = src_lm[:3].astype(np.float32)
    dst_pts = dst_lm[:3].astype(np.float32)
    return cv2.getAffineTransform(src_pts, dst_pts)


def _warp_face(
    src_bgr: np.ndarray,
    src_landmarks: np.ndarray,
    dst_bgr: np.ndarray,
    dst_landmarks: np.ndarray,
    dst_box: tuple[int, int, int, int],
    blend_strength: float = 1.0,
) -> np.ndarray:
    """Warp ``src_bgr`` face onto ``dst_bgr`` at the given box/landmarks.

    Steps:
    1. Compute affine from src landmarks → dst landmarks.
    2. Warp the entire src image into dst coordinate space.
    3. Build an elliptical mask around the dst face box.
    4. Use seamlessClone (or alpha blend if seamless fails) to merge.

    Returns:
        A copy of ``dst_bgr`` with the face swapped.
    """
    h, w = dst_bgr.shape[:2]
    M = _compute_affine_from_landmarks(src_landmarks, dst_landmarks)
    warped = cv2.warpAffine(src_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Build an elliptical mask centred on the dst face.
    x, y, fw, fh = dst_box
    cx = x + fw // 2
    cy = y + fh // 2
    # Expand slightly to cover forehead/chin.
    axes = (int(fw * 0.55), int(fh * 0.65))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    # Feather the mask edges a bit for smoother blending.
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, fw // 12))

    # seamlessClone expects a binary (0/255) mask and a centre point.
    # We'll use NORMAL_CLONE for natural lighting adaptation.
    try:
        result = cv2.seamlessClone(
            warped,
            dst_bgr,
            mask,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )
    except cv2.error:
        # Fallback to simple alpha blend if seamlessClone fails (e.g. mask
        # touches image border).
        alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        if blend_strength < 1.0:
            alpha = alpha * blend_strength
        result = (warped * alpha + dst_bgr * (1 - alpha)).astype(np.uint8)

    return result


def _load_reference_image(uploaded_file: Any) -> np.ndarray | None:
    """Decode an uploaded file into a BGR NumPy array."""
    if uploaded_file is None:
        return None
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        rgb = np.array(pil_image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logger.warning("참조 이미지 로드 실패: %s", exc)
        return None


@register
class FaceSwapFilter(BaseFilter):
    """Swap faces within an image or with an uploaded reference face."""

    name: str = "얼굴 교환 (Face Swap)"
    description: str = (
        "이미지 내 여러 얼굴을 서로 교환하거나, 업로드한 참조 이미지의 얼굴로 "
        "모든 얼굴을 대체합니다. YuNet landmark + seamlessClone 기반."
    )
    order: int = 420

    _MODES: ClassVar[dict[str, str]] = {
        "이미지 내 얼굴끼리 교환": "swap_within",
        "참조 이미지로 교체": "swap_reference",
    }

    def render_controls(self) -> dict[str, Any]:
        mode_label = st.sidebar.selectbox(
            "교환 모드",
            options=list(self._MODES.keys()),
            index=0,
            key="face_swap_mode",
        )
        mode = self._MODES[mode_label]

        reference_bgr: np.ndarray | None = None
        if mode == "swap_reference":
            st.sidebar.markdown("---")
            st.sidebar.markdown("**참조 얼굴 이미지**")
            ref_file = st.sidebar.file_uploader(
                "교환할 얼굴이 담긴 이미지를 업로드하세요",
                type=["png", "jpg", "jpeg", "webp"],
                key="face_swap_reference",
            )
            if ref_file is not None:
                reference_bgr = _load_reference_image(ref_file)
                if reference_bgr is not None:
                    # Show a small preview.
                    st.sidebar.image(
                        cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB),
                        caption="참조 이미지",
                        width=150,
                    )
            else:
                st.sidebar.info("참조 이미지를 업로드하면 모든 얼굴이 이 얼굴로 바뀝니다.")

        confidence = st.sidebar.slider(
            "검출 신뢰도",
            min_value=0.3,
            max_value=0.95,
            value=0.6,
            step=0.05,
            key="face_swap_confidence",
        )
        blend = st.sidebar.slider(
            "블렌딩 강도",
            min_value=0.3,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="1.0이면 완전히 교체, 낮으면 반투명하게 섞입니다.",
            key="face_swap_blend",
        )
        show_landmarks = st.sidebar.checkbox(
            "랜드마크 표시 (디버그)",
            value=False,
            key="face_swap_landmarks",
        )

        return {
            "mode": mode,
            "reference_bgr": reference_bgr,
            "confidence": float(confidence),
            "blend": float(blend),
            "show_landmarks": bool(show_landmarks),
        }

    def apply(
        self,
        image: np.ndarray,
        *,
        mode: str = "swap_within",
        reference_bgr: np.ndarray | None = None,
        confidence: float = 0.6,
        blend: float = 1.0,
        show_landmarks: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        faces = _get_yunet_landmarks(bgr, confidence_threshold=confidence)

        if mode == "swap_reference":
            return self._swap_with_reference(
                bgr, faces, reference_bgr, blend, show_landmarks
            )
        else:
            return self._swap_within_image(bgr, faces, blend, show_landmarks)

    def _swap_within_image(
        self,
        bgr: np.ndarray,
        faces: list[dict[str, Any]],
        blend: float,
        show_landmarks: bool,
    ) -> np.ndarray:
        """Swap faces with each other in round-robin fashion."""
        if len(faces) < 2:
            if show_landmarks and faces:
                bgr = self._draw_landmarks(bgr.copy(), faces)
            return bgr

        result = bgr.copy()
        n = len(faces)

        # Round-robin: face[i] gets face[(i+1) % n]'s appearance.
        for i in range(n):
            src_idx = (i + 1) % n
            src_face = faces[src_idx]
            dst_face = faces[i]

            result = _warp_face(
                bgr,  # always warp from original
                src_face["landmarks"],
                result,
                dst_face["landmarks"],
                dst_face["box"],
                blend_strength=blend,
            )

        if show_landmarks:
            result = self._draw_landmarks(result, faces)

        return result

    def _swap_with_reference(
        self,
        bgr: np.ndarray,
        faces: list[dict[str, Any]],
        reference_bgr: np.ndarray | None,
        blend: float,
        show_landmarks: bool,
    ) -> np.ndarray:
        """Replace all detected faces with the reference face."""
        if reference_bgr is None:
            if show_landmarks and faces:
                return self._draw_landmarks(bgr.copy(), faces)
            return bgr

        ref_faces = _get_yunet_landmarks(reference_bgr, confidence_threshold=0.4)
        if not ref_faces:
            st.warning("참조 이미지에서 얼굴을 찾지 못했습니다.")
            return bgr

        ref_face = ref_faces[0]  # Use the first (most confident) face.

        if not faces:
            return bgr

        result = bgr.copy()
        for dst_face in faces:
            result = _warp_face(
                reference_bgr,
                ref_face["landmarks"],
                result,
                dst_face["landmarks"],
                dst_face["box"],
                blend_strength=blend,
            )

        if show_landmarks:
            result = self._draw_landmarks(result, faces)

        return result

    @staticmethod
    def _draw_landmarks(
        canvas: np.ndarray,
        faces: list[dict[str, Any]],
    ) -> np.ndarray:
        """Overlay landmarks and bounding boxes for debugging."""
        colors = [
            (0, 255, 0),    # left eye - green
            (0, 255, 255),  # right eye - yellow
            (255, 0, 0),    # nose - blue
            (255, 0, 255),  # mouth left - magenta
            (255, 255, 0),  # mouth right - cyan
        ]
        for face in faces:
            x, y, w, h = face["box"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for idx, (px, py) in enumerate(face["landmarks"]):
                color = colors[idx % len(colors)]
                cv2.circle(canvas, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)
        return canvas
