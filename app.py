"""Streamlit front end for the vehicle counter (bonus feature).

Run with:
    streamlit run app.py

Everything the brief asks for is reachable from this page: pick an image or a
video, detect cars / motorcycles / buses / trucks, see boxes, labels and
confidence scores, read the total and per-class counts, optionally restrict
counting to a Region of Interest, and save the processed result.

It calls exactly the same pipeline functions as the CLI, so the two front ends
can never disagree about what a count means.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from vehicle_counter import annotate, config
from vehicle_counter.detector import VehicleDetector
from vehicle_counter.pipeline import process_image, process_video, source_kind
from vehicle_counter.roi import ROI, ROI_RULES

st.set_page_config(page_title="Parking Lot Vehicle Counter", page_icon="🚗", layout="wide")

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "input"
ROI_DIR = ROOT / "data" / "roi"
OUTPUT_DIR = ROOT / "outputs" / "ui"      # processed results are kept here
MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"]
MEDIA_EXTS = config.IMAGE_EXTS | config.VIDEO_EXTS
LINE_COLOR = "#3B82F6"
DRAW_WIDTH = 900          # on-screen width of the click-to-draw canvas


@st.cache_resource(show_spinner="Loading the YOLO model...")
def load_detector(model: str, device: str, conf: float, iou: float,
                  imgsz: int | str, class_ids: tuple[int, ...]) -> VehicleDetector:
    """Cached so the weights are not reloaded on every widget interaction.

    The arguments form the cache key, so changing any setting builds a fresh
    detector rather than silently reusing the old one.
    """
    return VehicleDetector(model_path=model, device=device, conf=conf,
                           iou=iou, imgsz=imgsz, class_ids=list(class_ids))


def first_frame(path: Path):
    """Representative frame, used for the ROI preview."""
    if source_kind(path) == "image":
        return cv2.imread(str(path))
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def bgr_to_rgb(frame):
    """OpenCV works in BGR; Streamlit expects RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def count_columns(counts: dict[str, int]) -> None:
    """One metric tile per class."""
    if not counts:
        st.info("No vehicles of the selected classes were found.")
        return
    for col, (name, count) in zip(st.columns(len(counts)), counts.items()):
        col.metric(name, count)


# ---------------------------------------------------------------------- #
# Sidebar - model and inference settings
# ---------------------------------------------------------------------- #
st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Weights", MODELS, index=0,
                                  help="Downloaded automatically on first use. "
                                       "n is fastest, m is most accurate.")
device = st.sidebar.selectbox("Device", [config.pick_device(), "cpu", "mps"], index=0)

st.sidebar.header("Detection")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, config.DEFAULT_CONF, 0.05,
                         help="Lower finds more vehicles but invents more false ones.")
iou = st.sidebar.slider("NMS IoU", 0.1, 0.9, config.DEFAULT_IOU, 0.05,
                        help="How much two boxes may overlap before one is suppressed.")
auto_imgsz = st.sidebar.checkbox("Auto inference size", value=True,
                                 help="Match the source resolution instead of shrinking "
                                      "everything to 640 px, which loses distant vehicles.")
imgsz: int | str = "auto"
if not auto_imgsz:
    imgsz = st.sidebar.select_slider("Inference size", [640, 960, 1280, 1600, 1920], value=1280)

class_names = st.sidebar.multiselect(
    "Count these classes",
    options=list(config.ALL_KNOWN_CLASSES.values()),
    default=list(config.VEHICLE_CLASSES.values()),
)
label_mode = st.sidebar.radio("Box captions", annotate.LABEL_MODES, index=0, horizontal=True,
                              help="auto shrinks captions to fit their box; "
                                   "none draws boxes only.")

st.sidebar.header("Video")
use_tracking = st.sidebar.checkbox("Track vehicles between frames", value=True)
min_track_hits = st.sidebar.number_input(
    "Frames before a track counts", 1, 30, 3,
    help="Guards the unique count against a tracker that keeps re-acquiring the "
         "same vehicle under a new id.",
)
stride = st.sidebar.number_input("Process every Nth frame", 1, 10, 1)
max_frames = st.sidebar.number_input("Max frames (0 = all)", 0, 10_000, 0)

name_to_id = {v: k for k, v in config.ALL_KNOWN_CLASSES.items()}
class_ids = tuple(sorted(name_to_id[n] for n in class_names))


# ---------------------------------------------------------------------- #
# Choose a source
# ---------------------------------------------------------------------- #
st.title("🚗 Parking Lot Vehicle Counter")
st.caption("Pre-trained YOLO11 + OpenCV. Detects cars, motorcycles, buses and trucks, "
           "draws boxes with confidence scores, and reports total and per-class counts - "
           "optionally only inside a region you choose.")

@st.cache_resource
def session_workdir() -> Path:
    """One scratch directory for uploads, reused across reruns.

    Streamlit re-executes this whole script on every widget interaction, so
    creating the directory inline would leak a new temp folder per click.
    """
    return Path(tempfile.mkdtemp(prefix="vehicle_counter_"))


samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in MEDIA_EXTS)
choices = ["Bundled sample", "Upload a file"] if samples else ["Upload a file"]
source_mode = st.radio("Source", choices, horizontal=True, label_visibility="collapsed")

source_path: Path | None = None
workdir = session_workdir()

if source_mode == "Bundled sample":
    picked = st.selectbox("Sample file", samples, format_func=lambda p: p.name)
    source_path = picked
else:
    uploaded = st.file_uploader(
        "Upload an image or a video",
        type=sorted(e.lstrip(".") for e in MEDIA_EXTS),
    )
    if uploaded:
        # Streamlit hands us bytes; OpenCV needs a real file on disk.
        source_path = workdir / uploaded.name
        source_path.write_bytes(uploaded.getbuffer())

if source_path is None:
    st.info("Upload a file, or run `python tools/fetch_samples.py` to get the bundled samples.")
    st.stop()

if not class_ids:
    st.warning("Pick at least one class to count.")
    st.stop()

kind = source_kind(source_path)
preview = first_frame(source_path)
if preview is None:
    st.error("Could not read that file.")
    st.stop()
height, width = preview.shape[:2]


# ---------------------------------------------------------------------- #
# Region of Interest
# ---------------------------------------------------------------------- #
st.subheader("Region of Interest")
roi_files = sorted(ROI_DIR.glob("*.json"))
roi_options = ["Whole frame", "Draw polygon", "Rectangle"]
if roi_files:
    roi_options.append("Saved polygon")
roi_options.append("Upload polygon JSON")

roi_mode = st.radio("Restrict counting to part of the frame?", roi_options,
                    horizontal=True, label_visibility="collapsed")

roi: ROI | None = None
roi_rule = "bottom"

if roi_mode == "Draw polygon":
    st.caption("**Click on the image** to drop a point. Three or more points make a "
               "region. Points are stored as fractions of the frame, so the same "
               "region still works if the footage changes resolution.")
    points: list[tuple[float, float]] = st.session_state.setdefault("draw_points", [])

    # Show the polygon built so far, drawn on the frame the user is clicking.
    canvas = annotate.draw_polygon_in_progress(
        preview.copy(), [(int(x * width), int(y * height)) for x, y in points]
    )
    click = streamlit_image_coordinates(
        bgr_to_rgb(canvas), width=DRAW_WIDTH, key="roi_canvas", cursor="crosshair",
    )

    # The component re-reports the same click on every rerun, so act only on a
    # click we have not already recorded.
    if click and click.get("unix_time") != st.session_state.get("last_click_time"):
        st.session_state["last_click_time"] = click.get("unix_time")
        points.append((click["x"] / click["width"], click["y"] / click["height"]))
        st.rerun()

    undo_col, clear_col, count_col = st.columns([1, 1, 3])
    if undo_col.button("Undo point", disabled=not points, width="stretch"):
        points.pop()
        st.rerun()
    if clear_col.button("Clear all", disabled=not points, width="stretch"):
        points.clear()
        st.rerun()
    count_col.write(
        f"**{len(points)} point(s)**" + ("" if len(points) >= 3 else " - need at least 3")
    )

    if len(points) >= 3:
        roi = ROI(points=list(points), name="drawn", normalized=True)

        save_col, name_col = st.columns([1, 2])
        roi_name = name_col.text_input("Save as", value="my_lot",
                                       label_visibility="collapsed",
                                       placeholder="name for this region")
        if save_col.button("Save region", width="stretch"):
            safe = "".join(c for c in roi_name if c.isalnum() or c in "-_") or "roi"
            saved = ROI(points=list(points), name=safe, normalized=True).save(
                ROI_DIR / f"{safe}.json"
            )
            st.success(f"Saved to `{saved.relative_to(ROOT)}` - it now appears under "
                       f"**Saved polygon**, and works from the CLI with "
                       f"`--roi {saved.relative_to(ROOT)}`")

elif roi_mode == "Rectangle":
    left, right = st.slider("Horizontal range", 0.0, 1.0, (0.0, 0.6), 0.01)
    top, bottom = st.slider("Vertical range", 0.0, 1.0, (0.15, 1.0), 0.01)
    if right > left and bottom > top:
        roi = ROI.from_rect(left, top, right, bottom, name="rectangle", normalized=True)
    else:
        st.warning("The rectangle has no area - widen the ranges.")
elif roi_mode == "Saved polygon":
    picked_roi = st.selectbox("ROI file", roi_files, format_func=lambda p: p.name)
    roi = ROI.load(picked_roi)
elif roi_mode == "Upload polygon JSON":
    roi_file = st.file_uploader("ROI JSON from tools/roi_picker.py", type=["json"], key="roi")
    if roi_file:
        roi_path = workdir / "roi.json"
        roi_path.write_bytes(roi_file.getbuffer())
        roi = ROI.load(roi_path)

if roi is not None:
    st.caption(f"ROI **{roi.name}** - {len(roi.points)} points")
    roi_rule = st.selectbox(
        "A vehicle counts when this point is inside the region", ROI_RULES, index=0,
        help="bottom = where the vehicle meets the ground (best for angled views); "
             "center = the box centroid; overlap = any part of the box.",
    )

if roi_mode != "Draw polygon":
    st.image(bgr_to_rgb(annotate.draw_roi(preview.copy(), roi)),
             caption=f"{source_path.name} - {width}x{height}"
                     + ("" if roi is None else f", ROI '{roi.name}' shaded"),
             width="stretch")


# ---------------------------------------------------------------------- #
# Run
# ---------------------------------------------------------------------- #
if not st.button("Run detection", type="primary"):
    st.stop()

detector = load_detector(model_name, device, conf, iou, imgsz, class_ids)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if kind == "image":
    with st.spinner("Detecting..."):
        result = process_image(source_path, detector, roi, roi_rule, OUTPUT_DIR,
                               save_json=True, label_mode=label_mode)

    st.subheader(f"{result.total} vehicles detected")
    count_columns(result.counts)
    st.image(bgr_to_rgb(cv2.imread(str(result.output_image))), width="stretch")
    st.caption(f"Inference took {result.elapsed_s * 1000:.0f} ms on {detector.device} "
               f"at imgsz {detector.imgsz_for(preview)}. "
               f"Saved to `{result.output_image.relative_to(ROOT)}`")

    with st.expander("Every detection (what YOLO actually returned)"):
        st.dataframe(
            pd.DataFrame([
                {
                    "class": d.cls_name,
                    "confidence": round(d.conf, 3),
                    "x1": round(d.xyxy[0]), "y1": round(d.xyxy[1]),
                    "x2": round(d.xyxy[2]), "y2": round(d.xyxy[3]),
                    "width": round(d.wh[0]), "height": round(d.wh[1]),
                }
                for d in sorted(result.detections, key=lambda d: -d.conf)
            ]),
            width="stretch", hide_index=True,
        )

    left_col, right_col = st.columns(2)
    left_col.download_button("Download annotated image", result.output_image.read_bytes(),
                             file_name=result.output_image.name, mime="image/jpeg")
    if result.summary_json:
        right_col.download_button("Download counts JSON", result.summary_json.read_bytes(),
                                  file_name=result.summary_json.name, mime="application/json")

else:
    bar = st.progress(0.0, text="Processing video...")

    def on_progress(done: int, total: int | None) -> None:
        if total:
            bar.progress(min(done / total, 1.0), text=f"Processing frame {done} of {total}")

    result = process_video(
        source_path, detector, roi, roi_rule, OUTPUT_DIR,
        use_tracking=use_tracking, min_track_hits=int(min_track_hits),
        stride=int(stride), max_frames=int(max_frames) or None,
        save_json=True, label_mode=label_mode,
        progress_cb=on_progress, show_progress=False,
    )
    bar.empty()

    # Occupancy and unique counts answer different questions - keep them apart.
    st.subheader("How full it gets (occupancy)")
    left_col, right_col = st.columns(2)
    left_col.metric("Peak vehicles in one frame", result.peak_occupancy)
    right_col.metric("Mean vehicles per frame", f"{result.mean_occupancy:.1f}")
    st.caption("Per-class counts on the busiest frame:")
    count_columns(result.peak_counts)

    if use_tracking:
        st.subheader("How many came through (unique vehicles)")
        left_col, right_col = st.columns(2)
        left_col.metric("Unique vehicles", result.unique_total,
                        help="Distinct track ids that survived the minimum-frames filter.")
        right_col.metric("Raw track ids", result.unique_raw,
                         help="Before filtering. The gap between these two is tracker ID churn.")
        count_columns(result.unique_by_class)

    st.video(result.output_video.read_bytes())
    st.caption(f"Processed {result.frames_processed} frames in {result.elapsed_s:.1f} s "
               f"({result.fps_processing:.1f} FPS) on {detector.device}. "
               f"Saved to `{result.output_video.relative_to(ROOT)}`")

    if result.per_frame:
        st.subheader("Vehicles in frame over time")
        chart = pd.DataFrame(result.per_frame)[["time_s", "occupancy"]].set_index("time_s")
        chart.columns = ["vehicles in frame"]
        st.line_chart(chart, color=LINE_COLOR, x_label="time (s)", y_label="vehicles")

    left_col, right_col = st.columns(2)
    left_col.download_button("Download annotated video", result.output_video.read_bytes(),
                             file_name=result.output_video.name, mime="video/mp4")
    if result.per_frame_csv:
        right_col.download_button("Download per-frame CSV", result.per_frame_csv.read_bytes(),
                                  file_name=result.per_frame_csv.name, mime="text/csv")
