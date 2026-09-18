#!/usr/bin/env python3
"""Draw a Region of Interest by clicking on the image, and save it as JSON.

The saved polygon is normalised to [0, 1], so the same file keeps working if
the scene is later supplied at a different resolution.

Usage:
    python tools/roi_picker.py --source data/input/parking_lot.jpg \
                               --out data/roi/main_lot.json

    # for a video, pick the frame to draw on
    python tools/roi_picker.py --source data/input/street_traffic.mp4 --frame 120

Controls:
    left click    add a point
    right click   undo the last point
    r             reset
    s             save and quit
    q / Esc       quit without saving
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vehicle_counter import annotate, config  # noqa: E402
from vehicle_counter.roi import ROI         # noqa: E402

WINDOW = "ROI picker  -  left: add   right: undo   r: reset   s: save   q: quit"
MAX_DISPLAY_WIDTH = 1400   # shrink oversized images so the window fits on screen


def load_frame(source: Path, frame_index: int) -> np.ndarray:
    """First (or Nth) frame of a video, or the image itself."""
    if source.suffix.lower() in config.VIDEO_EXTS:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {source}")
        if frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Could not read frame {frame_index} from {source}")
        return frame

    frame = cv2.imread(str(source))
    if frame is None:
        raise SystemExit(f"Could not read image: {source}")
    return frame


def render(base: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    """Draw the polygon in progress plus a short help banner."""
    canvas = annotate.draw_polygon_in_progress(base.copy(), points)

    status = f"{len(points)} point(s)" + ("" if len(points) >= 3 else "  - need at least 3")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"{status}   |   left: add   right: undo   r: reset   s: save   q: quit",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Click out an ROI polygon and save it as JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", required=True, help="image or video to draw on")
    parser.add_argument("--out", default="data/roi/roi.json", help="where to write the JSON")
    parser.add_argument("--frame", type=int, default=0, help="video only: frame to draw on")
    parser.add_argument("--name", default=None, help="ROI name (defaults to the output stem)")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    frame = load_frame(source, args.frame)
    full_h, full_w = frame.shape[:2]

    # Work on a shrunken copy if the image is larger than the screen, and scale
    # the clicked coordinates back up when saving.
    scale = min(1.0, MAX_DISPLAY_WIDTH / full_w)
    display = cv2.resize(frame, (int(full_w * scale), int(full_h * scale))) if scale < 1 else frame

    points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"Drawing on {source.name} at {display.shape[1]}x{display.shape[0]}"
          f"{'' if scale == 1 else f' (displayed at {scale:.0%})'}")

    try:
        while True:
            cv2.imshow(WINDOW, render(display, points))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                print("Cancelled, nothing written.")
                return 1
            if key == ord("r"):
                points.clear()
            if key == ord("s"):
                if len(points) < 3:
                    print("Need at least 3 points before saving.")
                    continue
                roi = ROI(
                    points=[(x / scale / full_w, y / scale / full_h) for x, y in points],
                    name=args.name or Path(args.out).stem,
                    normalized=True,
                )
                path = roi.save(args.out)
                print(f"Saved {len(points)}-point ROI to {path}")
                print(f"\nUse it with:\n  python detect_vehicles.py --source {source} --roi {path}")
                return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
