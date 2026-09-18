"""Thin wrapper around the pre-trained Ultralytics YOLO model.

This is the only module in the project that touches raw Ultralytics objects.
Everything downstream works with the plain `Detection` dataclass below, which
keeps the drawing and counting code independent of the library's API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_counter import config


@dataclass(frozen=True)
class Detection:
    """One detected vehicle in one frame.

    Attributes map directly onto what YOLO returns for a single box:
      cls_id   - COCO class id, e.g. 2 for "car"          (from boxes.cls)
      conf     - confidence in [0, 1] for that class      (from boxes.conf)
      xyxy     - box corners in absolute pixels           (from boxes.xyxy)
      track_id - stable id across frames, tracking only   (from boxes.id)
    """

    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        """Centroid of the box."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box.

        This approximates where the vehicle touches the ground, which is a much
        better "where is this car actually parked" point than the centroid when
        the camera looks down the lot at an angle.
        """
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def wh(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1, y2 - y1)

    @property
    def label(self) -> str:
        """Text drawn above the box, e.g. 'car 0.87' or '#12 car 0.87'."""
        prefix = f"#{self.track_id} " if self.track_id is not None else ""
        return f"{prefix}{self.cls_name} {self.conf:.2f}"


class VehicleDetector:
    """Loads a pre-trained YOLO model once and runs it on frames.

    No training happens anywhere in this project - the weights are downloaded
    ready-made by Ultralytics on first use.
    """

    def __init__(
        self,
        model_path: str = config.DEFAULT_MODEL,
        device: str | None = None,
        conf: float = config.DEFAULT_CONF,
        iou: float = config.DEFAULT_IOU,
        imgsz: int | str = config.DEFAULT_IMGSZ,
        class_ids: list[int] | None = None,
        tracker: str = config.DEFAULT_TRACKER,
    ) -> None:
        from ultralytics import YOLO  # imported lazily: it is a heavy import

        self.model_path = model_path
        self.device = config.pick_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker
        self.class_ids = sorted(class_ids or config.VEHICLE_CLASSES.keys())

        self.model = YOLO(model_path)
        # The model knows its own id -> name map; prefer it over our constants so
        # a non-COCO model would still label correctly.
        self.names: dict[int, str] = dict(self.model.names)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def imgsz_for(self, frame: np.ndarray) -> int:
        """Concrete inference size for this frame (see config.resolve_imgsz)."""
        return config.resolve_imgsz(self.imgsz, frame.shape[:2])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Single-image detection. No temporal state, so no track ids."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz_for(frame),
            device=self.device,
            # Filtering by class INSIDE the call is deliberate: ultralytics drops
            # non-vehicle classes before NMS, so a person standing next to a car
            # never competes with it for suppression.
            classes=self.class_ids,
            verbose=False,
        )
        return self._to_detections(results[0])

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Detection plus multi-object tracking.

        `persist=True` tells ultralytics that consecutive calls are consecutive
        frames of the same video, which is what lets it carry ids forward.
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz_for(frame),
            device=self.device,
            classes=self.class_ids,
            persist=True,
            tracker=self.tracker,
            verbose=False,
        )
        return self._to_detections(results[0])

    def reset_tracker(self) -> None:
        """Forget track ids so a new video starts numbering from scratch."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            if hasattr(tracker, "reset"):
                tracker.reset()

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #
    def _to_detections(self, result) -> list[Detection]:
        """Convert one ultralytics Result into our own Detection objects.

        `result.boxes` is a Boxes object holding parallel tensors:
          .xyxy (N, 4) float - box corners in pixels of the input frame
          .conf (N,)   float - confidence per box, already past conf/NMS filtering
          .cls  (N,)   float - class id per box
          .id   (N,)   float - track id per box, or None when not tracking
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        detections: list[Detection] = []
        for i in range(len(clss)):
            cls_id = int(clss[i])
            detections.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self.names.get(cls_id, str(cls_id)),
                    conf=float(confs[i]),
                    xyxy=tuple(float(v) for v in xyxy[i]),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return detections


def count_by_class(detections: list[Detection]) -> dict[str, int]:
    """Per-class tally, e.g. {'car': 12, 'truck': 2}. Sorted for stable output."""
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.cls_name] = counts.get(det.cls_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
