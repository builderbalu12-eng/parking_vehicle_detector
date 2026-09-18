"""Region of Interest support (bonus feature).

A parking lot camera usually sees more than the lot: a feeder road, a
neighbouring street, part of another property. Counting everything the model
finds therefore over-reports the lot's occupancy. An ROI is a polygon drawn once
over the area we actually care about; detections whose anchor point falls
outside it are dropped before counting.

Points may be stored either in absolute pixels or normalised to [0, 1]. Storing
them normalised means the same ROI file keeps working if the same scene is later
supplied at a different resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vehicle_counter.detector import Detection

# How a detection is tested against the polygon.
ROI_RULES = ("bottom", "center", "overlap")


@dataclass
class ROI:
    """A polygon region. Needs at least 3 points to enclose any area."""

    points: list[tuple[float, float]]
    name: str = "roi"
    normalized: bool = False
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(
                f"An ROI needs at least 3 points to enclose an area, got {len(self.points)}"
            )
        self.points = [(float(x), float(y)) for x, y in self.points]

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> "ROI":
        data = json.loads(Path(path).read_text())
        return cls(
            points=[tuple(p) for p in data["points"]],
            name=data.get("name", Path(path).stem),
            normalized=bool(data.get("normalized", False)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "normalized": self.normalized,
                    "points": [list(p) for p in self.points],
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def from_rect(
        cls, x1: float, y1: float, x2: float, y2: float, name: str = "rect", normalized: bool = True
    ) -> "ROI":
        """Axis-aligned rectangle. Used by the Streamlit slider controls."""
        return cls(
            points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            name=name,
            normalized=normalized,
        )

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def polygon(self, width: int, height: int) -> np.ndarray:
        """Polygon as an (N, 2) int32 array of pixel coordinates.

        Cached per frame size: a video calls this once per detection per frame,
        and rebuilding the array every time is pure waste.
        """
        key = (width, height)
        cached = self._cache.get(key)
        if cached is None:
            if self.normalized:
                pts = [(x * width, y * height) for x, y in self.points]
            else:
                pts = list(self.points)
            cached = np.array(pts, dtype=np.int32)
            self._cache[key] = cached
        return cached

    def contains(
        self, det: Detection, width: int, height: int, rule: str = "bottom"
    ) -> bool:
        """Is this detection inside the region?

        bottom  - test the bottom-centre of the box (default). Best for angled
                  views: it is roughly where the vehicle meets the tarmac.
        center  - test the centroid. Fine for top-down views.
        overlap - true if the box and the polygon intersect at all. The most
                  permissive rule; useful when vehicles straddle the boundary.
        """
        poly = self.polygon(width, height)

        if rule == "center":
            return _point_inside(poly, det.center)
        if rule == "bottom":
            return _point_inside(poly, det.anchor)
        if rule == "overlap":
            x1, y1, x2, y2 = det.xyxy
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), det.center]
            if any(_point_inside(poly, c) for c in corners):
                return True
            # Also catch the case of a small polygon sitting entirely inside a
            # large box, where no box corner is inside the polygon.
            return any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in poly)
        raise ValueError(f"Unknown ROI rule {rule!r}, expected one of {ROI_RULES}")


def _point_inside(polygon: np.ndarray, point: tuple[float, float]) -> bool:
    """>= 0 means inside or exactly on the edge."""
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def filter_detections(
    detections: list[Detection],
    roi: ROI | None,
    width: int,
    height: int,
    rule: str = "bottom",
) -> list[Detection]:
    """Keep only detections inside the ROI.

    `roi=None` means "the whole frame", so callers do not need a separate code
    path for the no-ROI case.
    """
    if roi is None:
        return detections
    return [d for d in detections if roi.contains(d, width, height, rule)]
