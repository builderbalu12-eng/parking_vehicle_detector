"""End-to-end processing for a single image or a video file.

Both the CLI (`detect_vehicles.py`) and the Streamlit app (`app.py`) call into
these two functions, so the two front ends can never drift apart.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from vehicle_counter import annotate, config
from vehicle_counter.detector import Detection, VehicleDetector, count_by_class
from vehicle_counter.roi import ROI, filter_detections


# ---------------------------------------------------------------------- #
# Results
# ---------------------------------------------------------------------- #
@dataclass
class ImageResult:
    source: Path
    output_image: Path
    counts: dict[str, int]
    total: int
    detections: list[Detection]
    elapsed_s: float
    summary_json: Path | None = None


@dataclass
class VideoResult:
    source: Path
    output_video: Path
    frames_processed: int
    elapsed_s: float
    fps_processing: float
    peak_counts: dict[str, int]          # per-class tally on the busiest frame
    peak_occupancy: int                  # most vehicles visible at once
    mean_occupancy: float
    unique_total: int                    # distinct vehicles that passed the min-hits filter
    unique_by_class: dict[str, int]
    unique_raw: int = 0                  # distinct track ids before filtering
    per_frame: list[dict] = field(default_factory=list)
    summary_json: Path | None = None
    per_frame_csv: Path | None = None


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def source_kind(path: str | Path) -> str:
    """'image' or 'video', decided by file extension."""
    ext = Path(path).suffix.lower()
    if ext in config.IMAGE_EXTS:
        return "image"
    if ext in config.VIDEO_EXTS:
        return "video"
    raise ValueError(
        f"Unsupported file type {ext!r}. "
        f"Images: {sorted(config.IMAGE_EXTS)}  Videos: {sorted(config.VIDEO_EXTS)}"
    )


def _fourcc(code: str) -> int:
    """FourCC code, working on both OpenCV 4 and 5."""
    if hasattr(cv2.VideoWriter, "fourcc"):
        return cv2.VideoWriter.fourcc(*code)
    return cv2.VideoWriter_fourcc(*code)  # type: ignore[attr-defined]


def _transcode_h264(src: Path) -> Path:
    """Re-encode to H.264 so the result plays in browsers and QuickTime.

    OpenCV usually writes `mp4v` (MPEG-4 Part 2), which most browsers refuse to
    play - the Streamlit video panel would just show a black box. If ffmpeg is
    unavailable we keep the original file rather than failing the run.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src

    dst = src.with_name(src.stem + "_h264.mp4")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",          # required for broad player support
        "-movflags", "+faststart",      # lets playback start before full download
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return src

    # Swap the re-encoded file into the original path so callers see one output.
    src.unlink(missing_ok=True)
    return dst.replace(src)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------- #
# Image
# ---------------------------------------------------------------------- #
def process_image(
    source: str | Path,
    detector: VehicleDetector,
    roi: ROI | None = None,
    roi_rule: str = "bottom",
    out_dir: str | Path = "outputs",
    save_json: bool = True,
    label_mode: str = "auto",
) -> ImageResult:
    """Detect, count and annotate a single image."""
    source = Path(source)
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    h, w = frame.shape[:2]
    resolved_imgsz = detector.imgsz_for(frame)
    started = time.perf_counter()
    detections = detector.detect(frame)
    detections = filter_detections(detections, roi, w, h, roi_rule)
    elapsed = time.perf_counter() - started

    counts = count_by_class(detections)
    total = len(detections)

    extra = [
        f"source: {source.name}",
        f"model: {Path(detector.model_path).name}  imgsz: {resolved_imgsz}",
    ]
    if roi is not None:
        extra.append(f"ROI: {roi.name} - counting inside only ({roi_rule})")

    annotated = annotate.annotate_frame(
        frame, detections, counts, total, roi, extra, label_mode
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_image = out_dir / f"{source.stem}_annotated.jpg"
    cv2.imwrite(str(output_image), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    summary_json = None
    if save_json:
        summary_json = _write_json(
            out_dir / f"{source.stem}_counts.json",
            {
                "source": str(source),
                "kind": "image",
                "model": detector.model_path,
                "device": detector.device,
                "conf": detector.conf,
                "iou": detector.iou,
                "imgsz": resolved_imgsz,
                "roi": roi.name if roi else None,
                "roi_rule": roi_rule if roi else None,
                "resolution": [w, h],
                "total_vehicles": total,
                "counts_by_class": counts,
                "inference_seconds": round(elapsed, 4),
                "detections": [
                    {
                        "class": d.cls_name,
                        "confidence": round(d.conf, 4),
                        "box_xyxy": [round(v, 1) for v in d.xyxy],
                    }
                    for d in detections
                ],
            },
        )

    return ImageResult(
        source=source,
        output_image=output_image,
        counts=counts,
        total=total,
        detections=detections,
        elapsed_s=elapsed,
        summary_json=summary_json,
    )


# ---------------------------------------------------------------------- #
# Video
# ---------------------------------------------------------------------- #
def process_video(
    source: str | Path,
    detector: VehicleDetector,
    roi: ROI | None = None,
    roi_rule: str = "bottom",
    out_dir: str | Path = "outputs",
    use_tracking: bool = True,
    min_track_hits: int = 3,
    stride: int = 1,
    max_frames: int | None = None,
    save_json: bool = True,
    label_mode: str = "auto",
    progress_cb=None,
    show_progress: bool = True,
) -> VideoResult:
    """Detect, count and annotate every (stride-th) frame of a video.

    Two different numbers are reported, because they answer different questions:

      occupancy       how many vehicles are visible in THIS frame. For a parking
                      lot this is the number that matters - "how full is it".
      unique_total    how many DISTINCT vehicles were seen across the whole clip,
                      counted via tracker ids. For a lot entrance this is the
                      number that matters - "how many came through".

    Summing per-frame counts would be meaningless: a car parked for 300 frames
    would be counted 300 times.

    `min_track_hits` guards the unique count against ID churn. When the camera
    moves or a vehicle is briefly occluded the tracker drops the box and
    re-acquires it under a fresh id, inflating the total. Requiring an id to
    survive a few frames before it counts discards those one-off ghosts; raise
    it for shakier footage, set it to 1 to see the unfiltered number.
    """
    source = Path(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    resolved_imgsz = config.resolve_imgsz(detector.imgsz, (height, width))

    # Dropping frames shortens the clip unless we slow the output fps to match.
    out_fps = max(1.0, src_fps / max(1, stride))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / f"{source.stem}_annotated.mp4"

    writer = cv2.VideoWriter(str(output_video), _fourcc("avc1"), out_fps, (width, height))
    if not writer.isOpened():  # avc1 is not always available in the OpenCV build
        writer = cv2.VideoWriter(str(output_video), _fourcc("mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open a video writer for {output_video}")

    if use_tracking:
        detector.reset_tracker()

    expected = total_frames // max(1, stride) if total_frames else None
    if max_frames:
        expected = min(expected or max_frames, max_frames)

    id_hits: dict[int, int] = {}      # track id -> how many frames it survived
    confirmed_ids: set[int] = set()   # ids that cleared min_track_hits
    id_to_class: dict[int, str] = {}
    per_frame: list[dict] = []
    occupancies: list[int] = []
    peak_occupancy = 0
    peak_counts: dict[str, int] = {}

    bar = tqdm(total=expected, unit="frame", disable=not show_progress, desc="Processing")
    started = time.perf_counter()
    frame_idx = 0
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if stride > 1 and frame_idx % stride != 0:
                frame_idx += 1
                continue

            detections = detector.track(frame) if use_tracking else detector.detect(frame)
            detections = filter_detections(detections, roi, width, height, roi_rule)

            for det in detections:
                if det.track_id is not None:
                    id_hits[det.track_id] = id_hits.get(det.track_id, 0) + 1
                    id_to_class[det.track_id] = det.cls_name
                    if id_hits[det.track_id] >= min_track_hits:
                        confirmed_ids.add(det.track_id)

            counts = count_by_class(detections)
            occupancy = len(detections)
            occupancies.append(occupancy)
            if occupancy > peak_occupancy:
                peak_occupancy, peak_counts = occupancy, counts

            timestamp = frame_idx / src_fps if src_fps else 0.0
            extra = [f"frame {frame_idx}  t={timestamp:5.1f}s"]
            if use_tracking:
                extra.append(f"unique so far: {len(confirmed_ids)}")
            if roi is not None:
                extra.append(f"ROI: {roi.name} ({roi_rule})")

            annotated = annotate.annotate_frame(
                frame, detections, counts, occupancy, roi, extra, label_mode
            )
            writer.write(annotated)

            per_frame.append(
                {
                    "frame": frame_idx,
                    "time_s": round(timestamp, 3),
                    "occupancy": occupancy,
                    **{f"count_{k}": v for k, v in counts.items()},
                }
            )

            processed += 1
            frame_idx += 1
            bar.update(1)
            if progress_cb:
                progress_cb(processed, expected)
            if max_frames and processed >= max_frames:
                break
    finally:
        bar.close()
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - started
    output_video = _transcode_h264(output_video)

    unique_by_class: dict[str, int] = {}
    for track_id in confirmed_ids:
        cls_name = id_to_class[track_id]
        unique_by_class[cls_name] = unique_by_class.get(cls_name, 0) + 1
    unique_by_class = dict(sorted(unique_by_class.items(), key=lambda kv: (-kv[1], kv[0])))

    mean_occupancy = float(np.mean(occupancies)) if occupancies else 0.0

    summary_json = None
    per_frame_csv = None
    if save_json:
        summary_json = _write_json(
            out_dir / f"{source.stem}_counts.json",
            {
                "source": str(source),
                "kind": "video",
                "model": detector.model_path,
                "device": detector.device,
                "conf": detector.conf,
                "iou": detector.iou,
                "imgsz": resolved_imgsz,
                "tracking": use_tracking,
                "min_track_hits": min_track_hits,
                "stride": stride,
                "roi": roi.name if roi else None,
                "roi_rule": roi_rule if roi else None,
                "resolution": [width, height],
                "source_fps": round(src_fps, 2),
                "frames_processed": processed,
                "processing_fps": round(processed / elapsed, 2) if elapsed else 0.0,
                "elapsed_seconds": round(elapsed, 2),
                "peak_occupancy": peak_occupancy,
                "peak_counts_by_class": peak_counts,
                "mean_occupancy": round(mean_occupancy, 2),
                "unique_vehicles_total": len(confirmed_ids) if use_tracking else None,
                "unique_vehicles_by_class": unique_by_class if use_tracking else None,
                "unique_track_ids_before_filter": len(id_hits) if use_tracking else None,
            },
        )

        per_frame_csv = out_dir / f"{source.stem}_per_frame.csv"
        fieldnames = sorted({k for row in per_frame for k in row})
        # Keep the two most useful columns first.
        for key in ("time_s", "frame"):
            if key in fieldnames:
                fieldnames.remove(key)
                fieldnames.insert(0, key)
        with per_frame_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for row in per_frame:
                w.writerow(row)

    return VideoResult(
        source=source,
        output_video=output_video,
        frames_processed=processed,
        elapsed_s=elapsed,
        fps_processing=processed / elapsed if elapsed else 0.0,
        peak_counts=peak_counts,
        peak_occupancy=peak_occupancy,
        mean_occupancy=mean_occupancy,
        unique_total=len(confirmed_ids),
        unique_by_class=unique_by_class,
        unique_raw=len(id_hits),
        per_frame=per_frame,
        summary_json=summary_json,
        per_frame_csv=per_frame_csv,
    )
