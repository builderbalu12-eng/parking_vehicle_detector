"""Shared defaults: which classes count as vehicles, how they are drawn, and
which compute device to use.

The class ids below are COCO ids. Every YOLO model shipped by Ultralytics that
was trained on COCO uses this same id -> name mapping, which is why we can hard
code the ids instead of looking them up by name at runtime.
"""

from __future__ import annotations

# COCO class id -> human readable name.
# These are the four classes the brief asks for.
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Defined but NOT enabled by default: a bicycle is not really a vehicle for
# parking-lot occupancy, but it is easy to switch on with --classes.
OPTIONAL_CLASSES: dict[int, str] = {
    1: "bicycle",
}

ALL_KNOWN_CLASSES: dict[int, str] = {**OPTIONAL_CLASSES, **VEHICLE_CLASSES}

# Per-class colours in BGR, because OpenCV is BGR rather than RGB.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (200, 200, 60),   # bicycle  - teal
    2: (80, 200, 80),    # car      - green
    3: (60, 180, 255),   # motorcyc - orange
    5: (230, 120, 60),   # bus      - blue
    7: (80, 80, 230),    # truck    - red
}
FALLBACK_COLOR: tuple[int, int, int] = (200, 200, 200)

# Model + inference defaults.
DEFAULT_MODEL = "yolo11n.pt"   # downloaded automatically on first run (~5.6 MB)
DEFAULT_CONF = 0.25            # minimum confidence to keep a detection
DEFAULT_IOU = 0.45             # NMS IoU threshold: how much overlap before two boxes merge
# Inference resolution. YOLO letterboxes every frame to this size, so it is the
# single biggest lever on whether small/distant vehicles are found at all: a
# 2048 px wide lot photo squeezed into 640 px turns parked cars into ~15 px
# blobs the model cannot resolve. "auto" matches the source resolution (snapped
# to a multiple of 32 and capped) instead of silently throwing detail away.
DEFAULT_IMGSZ = "auto"
IMGSZ_MIN = 640
IMGSZ_MAX = 1920               # beyond this the accuracy gain flattens out
IMGSZ_STEP = 32                # YOLO strides require a multiple of 32

# Tracker config shipped with ultralytics. ByteTrack is a good default because
# it also keeps low-confidence boxes alive, which helps with partly occluded
# parked cars.
DEFAULT_TRACKER = "bytetrack.yaml"

# Drawing.
ROI_COLOR: tuple[int, int, int] = (0, 215, 255)  # amber
ROI_ALPHA = 0.20                                 # translucency of the ROI fill
PANEL_ALPHA = 0.55                               # translucency of the count panel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def color_for(cls_id: int) -> tuple[int, int, int]:
    """Colour for a class id, falling back to grey for anything unexpected."""
    return CLASS_COLORS.get(cls_id, FALLBACK_COLOR)


def resolve_imgsz(requested: int | str, frame_shape: tuple) -> int:
    """Turn an --imgsz value into a concrete inference size.

    Anything other than "auto" is used as given. "auto" snaps the frame's long
    side up to a multiple of 32 and clamps it to [IMGSZ_MIN, IMGSZ_MAX].
    """
    if requested != "auto":
        return int(requested)

    import math

    long_side = max(frame_shape[0], frame_shape[1])
    snapped = int(math.ceil(long_side / IMGSZ_STEP) * IMGSZ_STEP)
    return max(IMGSZ_MIN, min(IMGSZ_MAX, snapped))


def pick_device(requested: str | None = None) -> str:
    """Choose the torch device.

    On this Mac that means Apple's Metal backend ("mps"). We import torch lazily
    so that importing this module stays cheap for code that only needs the
    constants above.
    """
    if requested:
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:  # torch missing or backend probe failed
        pass
    return "cpu"
