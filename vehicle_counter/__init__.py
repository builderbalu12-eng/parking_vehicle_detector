"""Vehicle detection and counting with a pre-trained YOLO model."""

from vehicle_counter.detector import Detection, VehicleDetector
from vehicle_counter.roi import ROI
from vehicle_counter.pipeline import process_image, process_video

__all__ = [
    "Detection",
    "VehicleDetector",
    "ROI",
    "process_image",
    "process_video",
]
