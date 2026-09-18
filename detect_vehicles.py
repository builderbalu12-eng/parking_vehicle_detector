#!/usr/bin/env python3
"""Detect and count vehicles in a parking-lot image or video.

Examples
--------
  # image
  python detect_vehicles.py --source data/input/parking_lot.jpg

  # video, counting only inside a region drawn with tools/roi_picker.py
  python detect_vehicles.py --source data/input/parking_lot.mp4 --roi data/roi/lot.json

  # be stricter, and look harder for small/distant vehicles
  python detect_vehicles.py --source data/input/parking_lot.jpg --conf 0.4 --imgsz 1280
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vehicle_counter import annotate, config
from vehicle_counter.detector import VehicleDetector
from vehicle_counter.pipeline import process_image, process_video, source_kind
from vehicle_counter.roi import ROI, ROI_RULES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Count vehicles in an image or video with a pre-trained YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, help="path to an image or video file")
    p.add_argument("--model", default=config.DEFAULT_MODEL,
                   help="YOLO weights; downloaded automatically on first use")
    p.add_argument("--conf", type=float, default=config.DEFAULT_CONF,
                   help="minimum confidence to keep a detection")
    p.add_argument("--iou", type=float, default=config.DEFAULT_IOU,
                   help="NMS IoU threshold (higher keeps more overlapping boxes)")
    p.add_argument("--imgsz", default=config.DEFAULT_IMGSZ,
                   help="inference resolution: 'auto' matches the source "
                        f"(capped at {config.IMGSZ_MAX}), or give a number like 640")
    p.add_argument("--device", default=None,
                   help="mps / cpu / 0 for CUDA. Auto-detected when omitted")
    p.add_argument("--classes", nargs="+", default=None,
                   help=f"class names to count, from {sorted(set(config.ALL_KNOWN_CLASSES.values()))}")
    p.add_argument("--roi", default=None, help="path to an ROI JSON file (bonus feature)")
    p.add_argument("--roi-rule", default="bottom", choices=ROI_RULES,
                   help="which part of a box must be inside the ROI")
    p.add_argument("--out", default="outputs", help="output directory")
    p.add_argument("--labels", default="auto", choices=annotate.LABEL_MODES,
                   help="box captions: auto shrinks them to fit, full always "
                        "writes 'class conf', none draws boxes only")
    p.add_argument("--no-track", action="store_true",
                   help="video only: disable tracking, so no unique-vehicle count")
    p.add_argument("--min-track-hits", type=int, default=3,
                   help="video only: frames a track id must survive before it "
                        "counts as a unique vehicle (guards against ID churn)")
    p.add_argument("--stride", type=int, default=1,
                   help="video only: process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None,
                   help="video only: stop after N processed frames")
    p.add_argument("--no-json", action="store_true", help="skip the JSON/CSV sidecar files")
    return p


def resolve_class_ids(names: list[str] | None) -> list[int]:
    """Map user-supplied class names back to COCO ids."""
    if not names:
        return sorted(config.VEHICLE_CLASSES)

    name_to_id = {v: k for k, v in config.ALL_KNOWN_CLASSES.items()}
    ids: list[int] = []
    for name in names:
        key = name.strip().lower()
        if key not in name_to_id:
            raise SystemExit(
                f"Unknown class {name!r}. Choose from: {sorted(name_to_id)}"
            )
        ids.append(name_to_id[key])
    return sorted(set(ids))


def print_counts(title: str, counts: dict[str, int], total: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not counts:
        print("  (no vehicles detected)")
    for name, count in counts.items():
        print(f"  {name:<12} {count:>4}")
    print(f"  {'TOTAL':<12} {total:>4}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Source not found: {source}\nRun: python tools/fetch_samples.py")

    try:
        kind = source_kind(source)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if args.roi and not Path(args.roi).exists():
        raise SystemExit(f"ROI file not found: {args.roi}\n"
                         f"Create one with: python tools/roi_picker.py --source {source}")
    roi = ROI.load(args.roi) if args.roi else None

    print(f"Source : {source}  ({kind})")
    print(f"Model  : {args.model}")

    detector = VehicleDetector(
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        class_ids=resolve_class_ids(args.classes),
    )
    print(f"Device : {detector.device}")
    print(f"imgsz  : {args.imgsz}")
    print(f"Classes: {[detector.names.get(c, c) for c in detector.class_ids]}")
    if roi:
        print(f"ROI    : {roi.name} ({len(roi.points)} points, rule={args.roi_rule})")

    if kind == "image":
        result = process_image(
            source, detector, roi, args.roi_rule, args.out,
            save_json=not args.no_json, label_mode=args.labels,
        )
        print_counts("Vehicles detected", result.counts, result.total)
        print(f"\nInference took {result.elapsed_s * 1000:.0f} ms")
        print(f"Annotated image : {result.output_image}")
        if result.summary_json:
            print(f"Counts JSON     : {result.summary_json}")
        return 0

    result = process_video(
        source,
        detector,
        roi,
        args.roi_rule,
        args.out,
        use_tracking=not args.no_track,
        min_track_hits=args.min_track_hits,
        stride=args.stride,
        max_frames=args.max_frames,
        save_json=not args.no_json,
        label_mode=args.labels,
    )

    print_counts(
        f"Busiest frame (peak occupancy)", result.peak_counts, result.peak_occupancy
    )
    print(f"\n  mean occupancy : {result.mean_occupancy:.1f} vehicles per frame")
    if not args.no_track:
        print_counts(
            f"Unique vehicles across the clip (track seen in >= {args.min_track_hits} frames)",
            result.unique_by_class,
            result.unique_total,
        )
        print(f"\n  raw track ids before filtering: {result.unique_raw}")
    print(
        f"\nProcessed {result.frames_processed} frames in {result.elapsed_s:.1f} s "
        f"({result.fps_processing:.1f} FPS)"
    )
    print(f"Annotated video : {result.output_video}")
    if result.summary_json:
        print(f"Summary JSON    : {result.summary_json}")
        print(f"Per-frame CSV   : {result.per_frame_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
