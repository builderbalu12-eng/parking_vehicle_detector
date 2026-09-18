"""Drawing: bounding boxes, labels, the ROI polygon and the count panel.

All sizes are derived from the frame height so that a 640 px sample image and a
1080p video both come out legible without per-source tuning.
"""

from __future__ import annotations

import cv2
import numpy as np

from vehicle_counter import config
from vehicle_counter.detector import Detection
from vehicle_counter.roi import ROI

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _metrics(height: int) -> tuple[float, int]:
    """(font_scale, line_thickness) proportional to the frame height."""
    scale = max(0.40, min(1.10, height / 1000.0))
    thickness = max(1, int(round(height / 600.0)))
    return scale, thickness


# Below this font scale text stops being readable, so we degrade instead.
MIN_LABEL_SCALE = 0.33

LABEL_MODES = ("auto", "full", "none")


def draw_detections(
    frame: np.ndarray, detections: list[Detection], label_mode: str = "auto"
) -> np.ndarray:
    """Draw one coloured box + label per detection. Modifies `frame` in place.

    A busy parking lot can hold 100+ boxes, and full-size captions on all of
    them hide the very vehicles they describe. `label_mode="auto"` shrinks each
    caption to its box, then degrades gracefully:

        car 0.62  ->  0.62  ->  (no text, colour still encodes the class)

    Use "full" to force complete captions, "none" for boxes only.
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unknown label mode {label_mode!r}, expected one of {LABEL_MODES}")

    h, w = frame.shape[:2]
    base_scale, thickness = _metrics(h)

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        color = config.color_for(det.cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if label_mode == "none":
            continue

        text, scale = det.label, base_scale
        if label_mode == "auto":
            fitted = _fit_label(det, x2 - x1, base_scale, thickness)
            if fitted is None:
                continue  # no room for readable text; the box colour still tells us the class
            text, scale = fitted

        label_thickness = max(1, int(round(thickness * scale / base_scale)))
        _draw_label(frame, text, x1, y1, color, scale, label_thickness, w, h)

    return frame


def _fit_label(
    det: Detection, box_width: int, base_scale: float, thickness: int
) -> tuple[str, float] | None:
    """Largest readable (text, font_scale) that fits `box_width`, or None."""
    for text in (det.label, f"{det.conf:.2f}"):
        width = cv2.getTextSize(text, FONT, base_scale, thickness)[0][0]
        if width <= 0:
            continue
        scale = base_scale if width <= box_width else base_scale * box_width / width
        if scale >= MIN_LABEL_SCALE:
            return text, min(scale, base_scale)
    return None


def _draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
    frame_w: int,
    frame_h: int,
) -> None:
    """Filled caption above the box, nudged to stay inside the frame."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    pad = max(2, thickness + 1)
    box_w = tw + 2 * pad
    box_h = th + baseline + 2 * pad

    # Prefer sitting above the box; flip inside it when there is no room on top.
    top = y - box_h
    if top < 0:
        top = y
    left = min(max(0, x), max(0, frame_w - box_w))
    bottom = min(frame_h, top + box_h)

    cv2.rectangle(frame, (left, top), (left + box_w, bottom), color, -1)
    cv2.putText(
        frame,
        text,
        (left + pad, top + th + pad),
        FONT,
        scale,
        _readable_text_color(color),
        thickness,
        cv2.LINE_AA,
    )


def _readable_text_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on light fills, white on dark ones, by perceived luminance."""
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_roi(frame: np.ndarray, roi: ROI | None) -> np.ndarray:
    """Translucent fill plus a solid outline for the region."""
    if roi is None:
        return frame

    h, w = frame.shape[:2]
    _, thickness = _metrics(h)
    poly = roi.polygon(w, h)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], config.ROI_COLOR)
    cv2.addWeighted(overlay, config.ROI_ALPHA, frame, 1 - config.ROI_ALPHA, 0, frame)
    cv2.polylines(frame, [poly], True, config.ROI_COLOR, thickness, cv2.LINE_AA)

    # Anchor the name to the LOWEST vertex: the count panel always occupies the
    # top-left, so labelling the topmost vertex would usually hide the text.
    scale = _metrics(h)[0] * 0.8
    text = f"ROI: {roi.name}"
    tw, th = cv2.getTextSize(text, FONT, scale, thickness)[0]
    anchor = poly[int(np.argmax(poly[:, 1]))]
    x = int(np.clip(anchor[0], 4, max(4, w - tw - 4)))
    y = int(np.clip(anchor[1] - 8, th + 4, h - 4))
    cv2.putText(frame, text, (x, y), FONT, scale, config.ROI_COLOR, thickness, cv2.LINE_AA)
    return frame


def draw_polygon_in_progress(
    frame: np.ndarray, points: list[tuple[int, int]], closed: bool | None = None
) -> np.ndarray:
    """Draw a half-finished ROI polygon: numbered vertices plus the edges so far.

    Shared by the desktop picker (tools/roi_picker.py) and the Streamlit app, so
    clicking out a region looks the same in both. Unlike `draw_roi` this accepts
    fewer than 3 points, because that is the state you are in while drawing.
    """
    if not points:
        return frame

    h = frame.shape[0]
    _, thickness = _metrics(h)
    radius = max(3, int(round(h / 200)))
    pts = np.array(points, dtype=np.int32)

    if closed is None:
        closed = len(points) >= 3

    if len(points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], config.ROI_COLOR)
        cv2.addWeighted(overlay, config.ROI_ALPHA, frame, 1 - config.ROI_ALPHA, 0, frame)

    if len(points) >= 2:
        cv2.polylines(frame, [pts], closed, config.ROI_COLOR, thickness, cv2.LINE_AA)

    for i, point in enumerate(points):
        cv2.circle(frame, point, radius, config.ROI_COLOR, -1)
        cv2.circle(frame, point, radius, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(i + 1), (point[0] + radius + 3, point[1] - radius - 3),
                    FONT, _metrics(h)[0] * 0.7, config.ROI_COLOR, thickness, cv2.LINE_AA)
    return frame


def draw_count_panel(
    frame: np.ndarray,
    counts: dict[str, int],
    total: int,
    extra: list[str] | None = None,
) -> np.ndarray:
    """Semi-transparent panel in the top-left with the total and per-class rows."""
    h, w = frame.shape[:2]
    scale, thickness = _metrics(h)
    title_scale = scale * 1.25

    lines: list[tuple[str, float, tuple[int, int, int]]] = [
        (f"TOTAL VEHICLES: {total}", title_scale, (255, 255, 255))
    ]
    for name, count in counts.items():
        cls_id = next((k for k, v in config.ALL_KNOWN_CLASSES.items() if v == name), -1)
        lines.append((f"  {name}: {count}", scale, config.color_for(cls_id)))
    for line in extra or []:
        lines.append((line, scale * 0.9, (215, 215, 215)))

    pad = max(8, int(h / 90))
    gap = max(6, int(h / 130))
    sizes = [cv2.getTextSize(t, FONT, s, thickness)[0] for t, s, _ in lines]
    panel_w = max(sz[0] for sz in sizes) + 2 * pad
    panel_h = sum(sz[1] for sz in sizes) + gap * (len(lines) - 1) + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, config.PANEL_ALPHA, frame, 1 - config.PANEL_ALPHA, 0, frame)
    cv2.rectangle(frame, (pad, pad), (pad + panel_w, pad + panel_h), (90, 90, 90), 1)

    y = pad * 2
    for (text, size, color), (tw, th) in zip(lines, sizes):
        y += th
        cv2.putText(frame, text, (pad * 2, y), FONT, size, color, thickness, cv2.LINE_AA)
        y += gap

    return frame


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    counts: dict[str, int],
    total: int,
    roi: ROI | None = None,
    extra: list[str] | None = None,
    label_mode: str = "auto",
) -> np.ndarray:
    """Full overlay for one frame, drawn on a copy of the input."""
    out = frame.copy()
    draw_roi(out, roi)
    draw_detections(out, detections, label_mode)
    draw_count_panel(out, counts, total, extra)
    return out
