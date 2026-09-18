# Complete Walkthrough — Setup, Running, and How Everything Works

A study guide for explaining this project end to end: from the pixels entering the
neural network, through the counting logic, to the browser UI.

Every number in this document was measured on this machine with this code — the
commands to reproduce them are included.

---

## Contents

- [Part 0 — Setup and run](#part-0--setup-and-run)
- [Part 1 — What the project does](#part-1--what-the-project-does)
- [Part 2 — The technology stack](#part-2--the-technology-stack)
- [Part 3 — How YOLO actually works](#part-3--how-yolo-actually-works)
- [Part 4 — The code, file by file](#part-4--the-code-file-by-file)
- [Part 5 — The image path, step by step](#part-5--the-image-path-step-by-step)
- [Part 6 — The video path and tracking](#part-6--the-video-path-and-tracking)
- [Part 7 — ROI: the geometry](#part-7--roi-the-geometry)
- [Part 8 — Drawing the annotations](#part-8--drawing-the-annotations)
- [Part 9 — The Streamlit UI](#part-9--the-streamlit-ui)
- [Part 10 — Measured results](#part-10--measured-results)
- [Part 11 — Limitations and why they exist](#part-11--limitations-and-why-they-exist)
- [Part 12 — Questions you may be asked](#part-12--questions-you-may-be-asked)
- [Part 13 — Glossary](#part-13--glossary)

---

# Part 0 — Setup and run

## 0.1 Requirements

| Need | Why |
|---|---|
| Python 3.10+ | Project developed on 3.14 |
| ~1.5 GB disk | PyTorch is the bulk of it |
| Internet (first run only) | Downloads packages, samples, and the model weights |
| ffmpeg *(optional)* | Re-encodes output video to H.264 so it plays in browsers |

## 0.2 Install

```bash
cd vehicle-counter

# create an isolated environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# download the sample image + video (~9 MB)
python tools/fetch_samples.py
```

`uv` works identically and is much faster if you have it: `uv venv && uv pip install -r requirements.txt`.

**On the first detection run** Ultralytics downloads `yolo11n.pt` (5.4 MB) automatically.
That one-off pause is normal. Nothing is ever trained.

## 0.3 The four commands to demonstrate

```bash
# 1. Image
python detect_vehicles.py --source data/input/parking_lot.jpg
#    -> 101 vehicles (96 car, 4 truck, 1 bus)

# 2. Image, counting only inside a chosen region
python detect_vehicles.py --source data/input/parking_lot.jpg --roi data/roi/main_lot.json
#    -> 47 vehicles (the other lot and the street are excluded)

# 3. Video, with tracking
python detect_vehicles.py --source data/input/street_traffic.mp4
#    -> peak 15 in one frame, 119 unique vehicles across the clip

# 4. The web UI
streamlit run app.py
#    -> http://localhost:8501
```

## 0.4 The one setting that matters most

```bash
python detect_vehicles.py --source data/input/parking_lot.jpg --imgsz 640
#    -> 9 vehicles instead of 101
```

If you remember one thing for the review, remember this one. Part 3.6 explains why.

## 0.5 Where the results go

```
outputs/
├── parking_lot_annotated.jpg        the picture with boxes drawn
├── parking_lot_counts.json          settings + totals + every box
├── street_traffic_annotated.mp4     annotated video (H.264)
├── street_traffic_counts.json       summary
└── street_traffic_per_frame.csv     occupancy for every frame
```

`examples/` holds a committed copy so a reviewer can see results without running anything.

---

# Part 1 — What the project does

**The task.** Count vehicles — car, motorcycle, bus, truck — in parking-lot images and
video, using a pre-trained YOLO model. No training.

**The shape of the solution.** The model is used as-is. All the actual engineering is
*around* the model:

```
   image / video
        |
        v
  [ 1. YOLO detects objects ]          <- pre-trained, not modified
        |
        v
  [ 2. keep only vehicle classes ]     <- done inside the model call
        |
        v
  [ 3. keep only what is in the ROI ]  <- optional, bonus feature
        |
        v
  [ 4. tracking, for video ]           <- gives each vehicle a stable ID
        |
        v
  [ 5. count + draw + save ]
```

**How each requirement is met:**

| Requirement | Where |
|---|---|
| Pre-trained YOLO | `detector.py` loads `yolo11n.pt`, downloaded ready-made |
| car / motorcycle / bus / truck | COCO ids 2, 3, 5, 7 in `config.py` |
| Boxes, labels, confidence | `annotate.py` |
| Total + per-class counts | `count_by_class()` in `detector.py`, displayed everywhere |
| Image and video, saved output | `pipeline.py` — `process_image()` / `process_video()` |
| **Bonus:** ROI | `roi.py` + `tools/roi_picker.py` + the UI's draw mode |
| **Bonus:** UI | `app.py` (Streamlit) |

---

# Part 2 — The technology stack

| Library | Version | What it does here |
|---|---|---|
| **Ultralytics** | 8.4.150 | Supplies the YOLO11 model, runs inference, handles NMS, provides the ByteTrack tracker |
| **PyTorch** | 2.14.0 | The deep-learning runtime. Holds the weights, runs the tensor maths on GPU/CPU |
| **OpenCV** | 5.0.0 | Reads/writes images and video, draws every rectangle and label, does the point-in-polygon test |
| **NumPy** | 2.5.3 | Array maths; the bridge format between OpenCV and PyTorch |
| **lap** | 0.5.13 | Solves the linear assignment problem the tracker uses to match detections to tracks |
| **Streamlit** | 1.63 | The web UI |
| **streamlit-image-coordinates** | 0.4.1 | Reports where you clicked on an image, for drawing the ROI |
| **tqdm** | — | Progress bar while processing video |
| **ffmpeg** | 8.0.1 | External tool. Re-encodes output video to H.264 |

**Why these.** Ultralytics is the reference implementation of YOLO and ships pre-trained
COCO weights, which is exactly what the brief asks for. OpenCV is the standard for video
I/O and drawing. Everything else is a dependency of those two, except Streamlit.

**Why an image is a NumPy array.** OpenCV loads a picture as an array of shape
`(height, width, 3)` with `uint8` values 0–255. The 3 channels are **B, G, R** — blue
first, not red. This trips people up constantly: that is why the UI calls
`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before showing anything in the browser.

---

# Part 3 — How YOLO actually works

This is the part you must be able to explain. Everything here was verified against the
actual model, not recalled from documentation.

## 3.1 One-stage detection

Older detectors (R-CNN family) worked in two stages: first propose regions that might
contain something, then classify each region. Slow.

**YOLO = "You Only Look Once."** One forward pass over the image produces every box and
every class score simultaneously. That is why it runs in real time.

## 3.2 The three parts of the network

```
input image
    |
 [ BACKBONE ]   convolutional layers that extract features,
    |           shrinking resolution while deepening meaning
    |           (edges -> textures -> wheels -> "car-ness")
    |
 [ NECK ]       fuses features across scales, so that both
    |           big and small objects are represented well
    |
 [ HEAD ]       outputs the actual box coordinates and class scores
    |
 raw predictions
```

`yolo11n` — the "nano" model used here — has **2,624,080 parameters**. That is tiny;
the largest YOLO11 variant has ~57 M. Smaller = faster but misses more.

## 3.3 Predictions happen at three scales

The head makes predictions on three grids, at **strides 8, 16 and 32**. Stride means
"one cell of this grid covers this many input pixels".

For a 640×640 input:

| Stride | Grid | Cells | Catches |
|---:|---:|---:|---|
| 8 | 80×80 | 6,400 | small objects |
| 16 | 40×40 | 1,600 | medium objects |
| 32 | 20×20 | 400 | large objects |
| | | **8,400 total** | |

**YOLO11 is anchor-free**: one prediction per cell, no pre-defined anchor box shapes.
Older YOLOs predicted 3 anchors per cell.

## 3.4 What comes out of the network

Verified directly:

```
raw output tensor: (1, 84, 8400)
                    |   |    |
                    |   |    +--- 8400 candidate predictions
                    |   +-------- 84 numbers each: 4 box + 80 class scores
                    +------------ batch size
```

**There is no objectness score.** YOLOv5 had one (a separate "is there anything here?"
channel, multiplied with the class score). **YOLOv8 and YOLO11 removed it.** The
confidence you see *is* the class score. Getting this right matters — many blog posts
still describe the old behaviour.

**How the 4 box numbers work.** Each cell predicts the **distance from itself to the
four edges** of the box (left, top, right, bottom). Internally the head actually emits
144 channels per cell — `4 × 16 + 80` — because each of the 4 distances is predicted as
a probability distribution over **16 bins** (this is *DFL*, Distribution Focal Loss).
Taking the weighted average of each distribution yields one precise distance, which is
why boxes land on sub-pixel boundaries rather than snapping to the grid.

## 3.5 From 8,400 predictions to a handful of boxes

```
8400 raw predictions
   |
   |  sigmoid on the 80 class channels -> scores in [0,1]
   |  confidence = highest class score;  class = which one
   v
   |  DROP anything below --conf (default 0.25)
   v
   |  DROP any class we did not ask for (classes=[2,3,5,7])
   v
   |  NMS at --iou (default 0.45): remove duplicate boxes
   v
   |  rescale coordinates from letterboxed space back to the original image
   v
final detections -> results[0].boxes
```

**NMS (Non-Maximum Suppression)** removes duplicates. The network usually fires several
neighbouring cells for the same car. NMS:

1. Sort all surviving boxes by confidence.
2. Take the highest-scoring box, keep it.
3. Delete every other box that overlaps it by more than the IoU threshold.
4. Repeat with what remains.

**IoU (Intersection over Union)** measures overlap:

```
          area of overlap
IoU =  ----------------------      0 = no overlap, 1 = identical
          area of union
```

The threshold is a genuine trade-off in a packed car park: **raise it** and near-duplicate
boxes survive (double counting); **lower it** and two genuinely adjacent cars get merged
into one (under counting).

**Why we pass `classes=[2,3,5,7]` into the model call rather than filtering afterwards:**
Ultralytics applies the class filter *before* NMS. If we filtered afterwards, a person
standing beside a car could suppress the car's box during NMS, and that car would vanish
from the count. Order matters.

## 3.6 Letterboxing — and why `--imgsz` dominates everything

The network needs a fixed input size. `--imgsz` sets it. The image is resized
**preserving aspect ratio**, then padded to a multiple of 32.

The sample lot photo is 2048×932. Here is what the network actually receives, and how
tall a typical parked car ends up:

| `--imgsz` | Network input | Scale | Car height | Vehicles found |
|---:|---:|---:|---:|---:|
| 640 | 320×640 | 0.31 | **17 px** | **9** |
| 960 | 448×960 | 0.47 | 26 px | 24 |
| 1280 | 608×1280 | 0.62 | 34 px | 39 |
| 1600 | 736×1600 | 0.78 | 43 px | 66 |
| **1920** | 896×1920 | 0.94 | **52 px** | **101** |

A detector needs roughly **20–30 px** of object to work with. At `imgsz 640` a car is
17 px tall — below the floor, so the model genuinely cannot see it. This is not a
threshold problem and no amount of lowering `--conf` fixes it; the information was
destroyed during the resize.

**This is why `--imgsz` defaults to `auto` in this project**: the frame's long side,
snapped to a multiple of 32, capped at 1920. On MPS the cost is ~0.02 s → ~0.04 s, which
is nothing for 9 → 101 detections.

*Reproduce:* `python detect_vehicles.py --source data/input/parking_lot.jpg --imgsz 640`

## 3.7 What a confidence score is — and is not

It is the model's score for that box being that class, after thresholding and NMS.

**It is not a calibrated probability.** A box marked `0.90` is *not* "90% likely to be a
car" in any frequentist sense. Neural networks are famously overconfident. Treat it as a
**ranking signal** — useful for comparing detections within one image and for setting a
cut-off — not as a real-world likelihood.

## 3.8 Reading `results[0].boxes`

One inference returns a `Results` object. `.boxes` holds parallel tensors, one row per
detection:

| Attribute | Shape | Meaning |
|---|---|---|
| `boxes.xyxy` | (N, 4) | `[x1, y1, x2, y2]` — top-left and bottom-right corners, **in pixels of the original image** |
| `boxes.conf` | (N,) | confidence for the assigned class |
| `boxes.cls` | (N,) | COCO class id, as a float |
| `boxes.id` | (N,) | tracker ID — only when tracking, otherwise `None` |

`xywh` (centre + size) is also available; this project uses `xyxy` because OpenCV's
`rectangle()` takes two corners, so no conversion is needed.

The COCO ids used here: **2 = car, 3 = motorcycle, 5 = bus, 7 = truck** (and 1 = bicycle,
available but off by default).

---

# Part 4 — The code, file by file

```
detect_vehicles.py        CLI entry point           177 lines
app.py                    Streamlit UI              313 lines
vehicle_counter/
├── config.py             settings + constants      105
├── detector.py           YOLO wrapper              185
├── roi.py                region geometry           150
├── annotate.py           all drawing               210
└── pipeline.py           image/video orchestration 413
tools/
├── fetch_samples.py      sample downloader         174
└── roi_picker.py         desktop ROI drawing tool  147
```

**The one architectural decision:** `detector.py` is the *only* file that touches raw
Ultralytics objects. It converts them immediately into a plain `Detection` dataclass.
Everything downstream — counting, drawing, ROI, the UI — works with that dataclass. If
Ultralytics changed its API tomorrow, one file would need editing.

## 4.1 `config.py` — the knobs

Holds the class ids, the per-class BGR colours, and the defaults. Two functions worth
knowing:

```python
def pick_device(requested=None):
    """mps on Apple Silicon, else CUDA, else cpu."""

def resolve_imgsz(requested, frame_shape):
    """'auto' -> the frame's long side, snapped to a multiple of 32,
       clamped to [640, 1920]. Anything else is used as given."""
```

`resolve_imgsz` is the code behind Part 3.6. The 32 comes from the network's largest
stride; the 1920 cap is because the gain from 1920 → 2048 measured just 2 extra
detections.

## 4.2 `detector.py` — the model wrapper

**The `Detection` dataclass** — one detected vehicle:

```python
@dataclass(frozen=True)
class Detection:
    cls_id: int                 # 2
    cls_name: str               # "car"
    conf: float                 # 0.87
    xyxy: tuple                 # (x1, y1, x2, y2) in pixels
    track_id: int | None        # 12, when tracking

    @property
    def center(self):  ...      # box centroid
    @property
    def anchor(self):  ...      # BOTTOM-CENTRE: where the car meets the ground
    @property
    def label(self):   ...      # "#12 car 0.87"
```

`anchor` exists specifically for the ROI test — see Part 7.

**`VehicleDetector`** loads the model once and exposes two methods:

```python
def detect(self, frame):   # single image, no memory between calls
def track(self, frame):    # video, persist=True so IDs carry across frames
```

Both pass `classes=self.class_ids` into the call, for the pre-NMS reason in Part 3.5.

**`_to_detections()`** is the ~15-line conversion from Ultralytics tensors to
`Detection` objects:

```python
xyxy  = boxes.xyxy.cpu().numpy()      # GPU tensor -> CPU -> NumPy
confs = boxes.conf.cpu().numpy()
clss  = boxes.cls.cpu().numpy().astype(int)
ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
```

`.cpu()` matters: on MPS/CUDA the results live in GPU memory and must be copied to the
CPU before NumPy can read them.

## 4.3 `pipeline.py` — the orchestration

Two public functions, `process_image()` and `process_video()`. Both the CLI and the UI
call these, so the two front ends can never disagree about what a count means.

## 4.4 `roi.py`, `annotate.py`

Covered in Parts 7 and 8.

## 4.5 `detect_vehicles.py` — the CLI

`argparse` for the flags, then: load ROI if asked → build the detector → dispatch to
image or video by file extension → print a counts table.

---

# Part 5 — The image path, step by step

```python
frame = cv2.imread(source)                              # (H, W, 3) BGR uint8
resolved_imgsz = detector.imgsz_for(frame)              # 'auto' -> 1920
detections = detector.detect(frame)                     # -> list[Detection]
detections = filter_detections(detections, roi, w, h)   # ROI, if any
counts = count_by_class(detections)                     # {'car': 96, ...}
annotated = annotate.annotate_frame(...)                # draw everything
cv2.imwrite(out_path, annotated)                        # save
```

**Counting is just `len()`.** There is no clever logic: one detection = one vehicle.
All the difficulty lives in deciding *which* detections are real and which are inside
the region.

`count_by_class` is six lines — a tally sorted by count descending:

```python
counts = {}
for det in detections:
    counts[det.cls_name] = counts.get(det.cls_name, 0) + 1
```

**What gets saved:** the annotated `.jpg`, plus a `_counts.json` recording every setting
used (model, device, conf, iou, resolved imgsz, ROI) alongside the totals and every
individual box. Recording the settings matters — a count is meaningless without them.

---

# Part 6 — The video path and tracking

## 6.1 The loop

```python
cap = cv2.VideoCapture(source)
writer = cv2.VideoWriter(out, fourcc, out_fps, (width, height))

while True:
    ok, frame = cap.read()
    if not ok: break                       # end of file
    if stride > 1 and frame_idx % stride:  # optional frame skipping
        frame_idx += 1; continue

    detections = detector.track(frame)     # or .detect() if --no-track
    detections = filter_detections(...)    # ROI
    ...
    writer.write(annotated)
```

**Frame skipping keeps real time.** With `--stride 3` we process every 3rd frame, so the
output is written at `fps / 3`. A 16.6 s clip stays 16.6 s rather than speeding up 3×.
Verified: 498 frames @ 30 fps → 166 frames @ 10 fps = 16.6 s.

## 6.2 The two counts — the conceptual heart of the project

**You cannot sum per-frame counts.** A car parked for 300 frames would be counted 300
times. So there are two genuinely different questions:

| | Question | How |
|---|---|---|
| **Occupancy** | "How full is the lot right now?" | `len(detections)` in one frame |
| **Unique** | "How many vehicles came through?" | count of distinct track IDs |

For a **parking lot**, occupancy is the number you want. For a **lot entrance**, unique
is. Both are reported, plus mean and peak occupancy.

## 6.3 How tracking works (ByteTrack)

Detection is memoryless — it has no idea the car in frame 2 is the car from frame 1. The
tracker supplies that continuity.

**Per frame:**

```
1. PREDICT   A Kalman filter moves each existing track forward, estimating where
             that vehicle should now be, from its recent position and velocity.

2. MATCH (stage 1)   High-confidence detections (>= 0.25) are matched against those
             predictions. Cost = 1 - IoU. The assignment is solved optimally by
             the `lap` library (linear assignment / Hungarian algorithm) --
             this is why `lap` is a dependency.

3. MATCH (stage 2)   ByteTrack's signature idea: the LOW-confidence detections
             (>= 0.1) that would normally be discarded get a second chance to
             match the still-unmatched tracks. A partly occluded car often drops
             to a low score -- this recovers it instead of losing the ID.

4. BIRTH     Unmatched detections above 0.25 start a new track with a new ID.

5. DEATH     Unmatched tracks are kept "lost" for 30 frames (`track_buffer`)
             before being deleted, so a brief occlusion does not kill the ID.
```

Ultralytics' actual `bytetrack.yaml` defaults:

```yaml
track_high_thresh: 0.25   # stage-1 matching threshold
track_low_thresh:  0.1    # stage-2 threshold
new_track_thresh:  0.25   # score needed to start a new track
track_buffer:      30     # frames a lost track survives
match_thresh:      0.8    # max cost to accept a match
```

`persist=True` in our `track()` call is what tells Ultralytics these frames are
consecutive frames of one video, rather than unrelated images.

## 6.4 ID churn — and the fix

On the 17-second street clip the tracker produced **263 distinct IDs**, while the busiest
single frame held only **15 vehicles**.

**Why.** The camera is moving (dashcam). The Kalman filter assumes roughly constant
velocity, but when the *camera* moves, every box jumps in a way the filter did not
predict. IoU between prediction and detection falls below `match_thresh`, the track is
declared lost, and the same physical car is re-born under a new ID.

**Evidence** — track length distribution:

| Track must survive | Unique vehicles kept |
|---:|---:|
| 1 frame | 263 |
| 2 frames | 263 |
| **3 frames** | **119** |
| 8 frames | 75 |
| 20 frames | 55 |

The **median track lasted 2 frames**: 144 IDs existed for exactly two frames, then
vanished. Those are ghosts, not vehicles.

**The fix** — `--min-track-hits` (default 3). An ID only counts as a real vehicle once
it has survived 3 frames:

```python
id_hits[det.track_id] += 1
if id_hits[det.track_id] >= min_track_hits:
    confirmed_ids.add(det.track_id)
```

263 → 119. **Both numbers are reported**, because the gap between them is itself the
honest measure of how much to trust the figure.

## 6.5 Writing the video — the codec trap

OpenCV usually writes `mp4v` (MPEG-4 Part 2). **Most browsers refuse to play it** — the
Streamlit video panel would show a black box. So after writing, if `ffmpeg` is present,
the file is re-encoded:

```bash
ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart out.mp4
```

- `libx264` — H.264, which browsers do play
- `yuv420p` — the pixel format with the widest player support
- `+faststart` — moves the index to the front so playback can begin before full download

If ffmpeg is missing the original is kept rather than failing the run.

---

# Part 7 — ROI: the geometry

## 7.1 The problem

A parking-lot camera sees more than the lot: a feeder road, a neighbouring property, a
public street. Counting everything over-reports occupancy. On the sample image this is
the difference between **101** (everything) and **47** (the main lot only).

## 7.2 Storage format

```json
{
  "name": "main_lot",
  "normalized": true,
  "points": [[0.005, 0.16], [0.58, 0.16], [0.58, 0.45],
             [0.50, 0.62], [0.50, 0.99], [0.005, 0.99]]
}
```

**Normalised** means coordinates are fractions of width/height rather than pixels. The
same file therefore works if the same scene is later supplied at a different resolution —
1080p footage and a 4K still of the same car park share one ROI file.

Converting to pixels is cached per frame size, because a video asks for it once per
detection per frame:

```python
def polygon(self, width, height):
    key = (width, height)
    if key not in self._cache:
        pts = [(x * width, y * height) for x, y in self.points] if self.normalized else self.points
        self._cache[key] = np.array(pts, dtype=np.int32)
    return self._cache[key]
```

## 7.3 Is the point inside? — `cv2.pointPolygonTest`

```python
cv2.pointPolygonTest(polygon, (x, y), False) >= 0
```

Returns +1 inside, 0 exactly on the edge, −1 outside (`False` = don't compute the actual
distance, which is faster). Internally this is the classic **ray-casting** test: shoot a
ray from the point and count how many polygon edges it crosses — odd means inside, even
means outside. It works for any simple polygon, convex or concave.

## 7.4 *Which* point to test — the interesting decision

A box is a rectangle; the region is a polygon. Which part of the box must be inside?

```
        +---------------+
        |               |
        |   [ car ]     |
        |       x       |  <-- center: the centroid
        |               |
        +-------o-------+  <-- bottom: where the tyres meet the ground
```

**`bottom` is the default.** In an angled view — which is what a lot camera gives you —
a car's *centroid* floats in the air above the tarmac, often over the **neighbouring
bay**, while the bottom-centre sits in the bay the car actually occupies.

All three rules are available, and their behaviour on the sample confirms the logic:

| `--roi-rule` | Tests | Count |
|---|---|---:|
| `bottom` | bottom-centre of the box | **47** |
| `center` | centroid | 49 |
| `overlap` | any part of the box touches the region | 50 |

Exactly the expected ordering — `overlap` is the most permissive, `bottom` the strictest.

## 7.5 The no-ROI case

```python
def filter_detections(detections, roi, width, height, rule="bottom"):
    if roi is None:
        return detections                      # "the whole frame"
    return [d for d in detections if roi.contains(d, width, height, rule)]
```

`roi=None` means the whole frame, so there is no separate code path for "no ROI" —
one function handles both.

---

# Part 8 — Drawing the annotations

All in `annotate.py`. Everything scales with frame height, so a 640 px image and a 1080p
video are both legible without per-source tuning:

```python
def _metrics(height):
    scale = max(0.40, min(1.10, height / 1000.0))
    thickness = max(1, int(round(height / 600.0)))
    return scale, thickness
```

## 8.1 Boxes and labels

Per detection: one `cv2.rectangle` in the class colour, then a filled caption above it.
Text colour is chosen by **perceived luminance** so it stays readable on any box colour:

```python
luminance = 0.114*b + 0.587*g + 0.299*r      # the eye is most sensitive to green
return (0,0,0) if luminance > 140 else (255,255,255)
```

## 8.2 The label problem, and `--labels auto`

The sample lot has **101 boxes**. Full-size captions on all of them hid the very cars
they described. So captions degrade gracefully — each is shrunk to fit its own box, and
if even that is unreadable it falls back:

```
car 0.62   ->   0.62   ->   (no text; the box colour still encodes the class)
```

That is `--labels auto`. `full` forces complete captions, `none` draws boxes only.

## 8.3 Translucent overlays

The ROI shading and the count panel are drawn by compositing:

```python
overlay = frame.copy()
cv2.fillPoly(overlay, [poly], ROI_COLOR)
cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)   # 20% tint
```

`addWeighted` is a per-pixel blend: `result = a*α + b*(1-α)`.

Small detail worth mentioning: the ROI's name is drawn at the polygon's **lowest** vertex,
because the count panel always occupies the top-left and would otherwise hide it.

---

# Part 9 — The Streamlit UI

## 9.1 Streamlit's execution model — the thing to understand

**Streamlit re-runs the entire script, top to bottom, on every single interaction.**
Move a slider, and the whole file executes again. There is no callback/event model.

Three consequences shape `app.py`:

**1. The model must be cached**, or the weights reload on every click:

```python
@st.cache_resource(show_spinner="Loading the YOLO model...")
def load_detector(model, device, conf, iou, imgsz, class_ids):
    return VehicleDetector(...)
```

The arguments form the cache key — change the confidence slider and you get a *new*
detector rather than a stale one.

**2. State that must survive reruns goes in `st.session_state`** — that is where the
polygon points you click are kept.

**3. Anything created inline is recreated every rerun.** The upload scratch directory is
cached for exactly this reason; creating it inline leaked a temp folder per click.

## 9.2 What the UI offers

- **Source** — a bundled sample or your own upload
- **Sidebar** — model, device, confidence, IoU, inference size, which classes, caption mode; and for video: tracking on/off, min track hits, frame stride, max frames
- **ROI** — whole frame / **draw polygon** / rectangle sliders / saved region / uploaded JSON
- **Results** — annotated output, total and per-class tiles, a table of every raw detection, occupancy-over-time chart for video, and download buttons
- Outputs are also written to `outputs/ui/`

## 9.3 Click-to-draw polygons — how it works

A Streamlit *custom component* is a small web page in an iframe that can send a value
back to Python. `streamlit-image-coordinates` reports where you clicked on an image.

Its JavaScript sends:

```js
{ x: offsetX, y: offsetY, width: img.width, height: img.height, unix_time: ... }
```

Crucially, `img.width`/`img.height` are the **displayed** dimensions (the component sets
them itself from the natural aspect ratio), not the original ones. So normalising is one
line — and gives exactly the fraction-based format the ROI file wants:

```python
points.append((click["x"] / click["width"], click["y"] / click["height"]))
```

**The deduplication detail:** because the script re-runs constantly, the component
re-reports the *same* click every time. Without a guard, one click would add infinite
points. Hence:

```python
if click and click.get("unix_time") != st.session_state.get("last_click_time"):
    st.session_state["last_click_time"] = click.get("unix_time")
    points.append(...)
    st.rerun()
```

**Verification.** Simulated clicks tracing the known `main_lot` region round-tripped with
**0.0 error**, and running the drawn polygon gave **47 vehicles** — identical to the CLI
with the hand-written JSON.

The polygon preview is drawn by `annotate.draw_polygon_in_progress()`, which is **shared
with the desktop picker** `tools/roi_picker.py`, so both look identical.

## 9.4 The desktop alternative

`tools/roi_picker.py` opens an OpenCV window: left click adds a point, right click undoes,
`r` resets, `s` saves, `q` quits. It scales large images down to fit the screen and maps
clicks back to original coordinates.

---

# Part 10 — Measured results

All on Apple Silicon, `mps` backend, `yolo11n.pt`.

**Sample image** (2048×932 parking lot) — **101 vehicles**: 96 car, 4 truck, 1 bus.
With the `main_lot` ROI: **47**.

**Sample video** (1226×370, 498 frames, 16.6 s): peak occupancy **15**, mean **8.1**,
unique **119** (raw 263). Processed at **41 FPS** — faster than real time.

**Inference size** (Part 3.6): 9 → 101 detections from 640 → 1920, costing 0.02 s → 0.04 s.

**Confidence threshold:**

| `--conf` | total | car | truck | bus |
|---:|---:|---:|---:|---:|
| 0.10 | 135 | 122 | 12 | 1 |
| 0.15 | 123 | 115 | 7 | 1 |
| **0.25** | **101** | 96 | 4 | 1 |
| 0.40 | 72 | 70 | 2 | 0 |
| 0.60 | 34 | 34 | 0 | 0 |

There is no threshold at which the count is simply "correct".

**Device:**

| Device | Per 1920 px image | Video |
|---|---:|---:|
| `mps` | 41 ms | 41 FPS |
| `cpu` | 200 ms | — |

MPS is ~**5× faster**.

---

# Part 11 — Limitations and why they exist

## 11.1 Top-down views break it — the bundled failure case

`topdown_lot.mp4` (fetch with `--all`) is included **on purpose** to demonstrate this.

| Setting | Frames with any detection | Detections |
|---|---:|---:|
| defaults | 28 of 377 | 27 car, 2 bus, 1 truck |
| `--conf 0.10 --imgsz 1920` | 69 of 377 | 51 car, 12 truck, **35 bus** |

**Why.** COCO photographs are overwhelmingly taken at ground level. A camera pointed
straight down shows a car silhouette the model has essentially never seen. Lowering the
threshold does not recover the vehicles — it **invents 35 buses** in a lot containing
none.

This is a **data mismatch**, not a tuning problem, and it is the single most important
limitation to be honest about. The real fix is fine-tuning on overhead imagery
(DOTA / VisDrone style), which the brief excludes.

The sample lot photo works well precisely because it is elevated but *oblique* — still
close to COCO's viewpoint.

## 11.2 Class confusion distorts the per-class split

`car` vs `truck` is the weak boundary: SUVs, vans and pickups sit between the two COCO
definitions and flip with angle, colour and occlusion. In the table above, `truck` moves
from 12 to 2 purely by changing the threshold — the vehicles obviously did not change.

**The total is considerably more trustworthy than the per-class breakdown.**

## 11.3 Occlusion under-counts

In a packed lot cars hide one another. A half-hidden car may fall below the threshold,
and NMS may merge two adjacent cars into one box. Both push the count **down** — so in a
full lot this tool is more likely to under-count than over-count.

## 11.4 Others

- **Small/distant vehicles** are still missed even at 1920. Tiled inference (SAHI) would help.
- **The ROI is image-space.** It survives a resolution change but *not* the camera moving, panning or zooming. No homography or calibration.
- **No empty-bay awareness.** This counts vehicles, not spaces. It cannot say "12 bays free" without a map of the bays.
- **ID churn** is mitigated by `--min-track-hits`, not eliminated.
- **Only `yolo11n`** was measured — the smallest model. `--model yolo11s.pt` detects more, slower.
- **Daylight only.** Night, rain and glare are untested.
- **No unit tests.** Verification was manual and visual.

---

# Part 12 — Questions you may be asked

**Q. Did you train the model?**
No — the brief rules it out. `yolo11n.pt` is downloaded pre-trained on COCO, which
already contains car, motorcycle, bus and truck. All the engineering is around the
model: class filtering, counting semantics, the ROI, and presentation.

**Q. What does the confidence score mean?**
The model's score that this box is this class, after thresholding and NMS. In YOLOv8/11
it *is* the class score — there is no separate objectness channel any more, unlike
YOLOv5. It is not a calibrated probability; it is a ranking signal.

**Q. What exactly does YOLO output?**
For a 640×640 input, a tensor of `(1, 84, 8400)`: 8400 candidate predictions (80×80 +
40×40 + 20×20 grid cells at strides 8/16/32), each with 4 box numbers and 80 class
scores. Those get thresholded, class-filtered and NMS'd down to the final handful.

**Q. How is the box encoded?**
Anchor-free: each cell predicts the distance to the four edges of the box. Internally
each distance is a 16-bin distribution (DFL) whose expected value is taken, which gives
sub-pixel precision. Ultralytics hands it to us as `xyxy` pixel corners.

**Q. What is NMS and what does `--iou` do?**
Non-Maximum Suppression removes duplicate boxes for the same object: keep the highest-
confidence box, delete anything overlapping it by more than the IoU threshold, repeat.
In a packed lot it is a real trade-off — too high double-counts, too low merges adjacent
cars.

**Q. Why filter classes inside the model call?**
Ultralytics applies `classes=` before NMS. If I filtered after, a pedestrian beside a car
could suppress the car's box and that car would disappear from the count.

**Q. Your counts changed a lot with one flag. Why?**
`--imgsz`. The model resizes every input to a fixed size. At 640, a car in this 2048 px
photo becomes 17 px tall — below the ~20–30 px a detector needs, so the information is
destroyed before the network ever sees it. At 1920 the same car is 52 px and the count
goes from 9 to 101. That is why the default is `auto`.

**Q. How do you count vehicles in a video without double counting?**
I never sum per-frame counts. I report two separate numbers: occupancy (how many are
visible in one frame — the right figure for "how full is the lot") and unique vehicles
(distinct tracker IDs across the clip — the right figure for "how many came through").

**Q. How does the tracking work?**
ByteTrack. A Kalman filter predicts where each existing track should be; detections are
matched to those predictions by IoU, solved optimally with a linear-assignment algorithm
(the `lap` dependency). ByteTrack's distinctive idea is a second matching pass using the
*low*-confidence detections, which recovers partly occluded vehicles instead of losing
their ID.

**Q. Why is the unique count 119 and not 263?**
263 is the raw number of tracker IDs, and it is inflated: the camera moves, so the
tracker keeps losing and re-acquiring the same car under a new ID. The median track
lasted 2 frames. Requiring an ID to survive 3 frames discards those ghosts and gives 119.
I report both, because the gap shows how much to trust the number.

**Q. Why bottom-centre for the ROI test?**
In an angled view the box centroid floats above the tarmac and often sits over the
neighbouring bay, while the bottom-centre is roughly where the tyres meet the ground —
the bay the car actually occupies. `center` and `overlap` are also available.

**Q. Where does this fail?**
Top-down cameras, and I ship a video that proves it: detections in only 28 of 377 frames,
and pushing the threshold down invents 35 buses that do not exist. COCO is shot at ground
level, so this is a data mismatch no parameter fixes. Also: per-class splits are shaky
(SUV/van/pickup flip between car and truck), dense occlusion under-counts, and the ROI
breaks if the camera moves.

**Q. How would you improve it?**
In order: (1) fine-tune on overhead parking imagery — fixes the biggest failure by a wide
margin; (2) tiled/SAHI inference for small distant vehicles; (3) map the actual bays so
the output becomes "17 of 60 free" rather than a raw count; (4) calibrate the confidence
threshold against a hand-labelled subset instead of picking it by eye.

**Q. Why Streamlit and not Flask?**
The brief suggested Streamlit or Gradio, and for a data app it is far less code — no
routes, templates or JavaScript. The cost is its rerun-everything execution model, which
is why the model is cached and the drawn points live in `session_state`.

**Q. How do the CLI and UI stay consistent?**
They both call the same two functions in `pipeline.py`. No detection or counting logic
exists in `app.py` at all.

---

# Part 13 — Glossary

| Term | Meaning |
|---|---|
| **COCO** | Common Objects in Context — the 80-class dataset YOLO is pre-trained on |
| **Inference** | Running a trained model to get predictions (as opposed to training) |
| **Backbone / Neck / Head** | Feature extraction / multi-scale fusion / final prediction layers |
| **Stride** | How many input pixels one grid cell covers (8, 16, 32 here) |
| **Anchor-free** | One prediction per grid cell, with no pre-defined box shapes |
| **DFL** | Distribution Focal Loss — box edges predicted as 16-bin distributions for sub-pixel precision |
| **Objectness** | An "is anything here?" score. Present in YOLOv5, **removed** in v8/v11 |
| **NMS** | Non-Maximum Suppression — removes duplicate boxes for one object |
| **IoU** | Intersection over Union — overlap ratio between two boxes, 0 to 1 |
| **Letterboxing** | Aspect-preserving resize plus padding, to reach the network's input size |
| **Confidence** | The model's class score for a box; a ranking signal, not a calibrated probability |
| **xyxy / xywh** | Two box formats: corner-corner, or centre-plus-size |
| **Tracking** | Linking detections of the same object across frames, giving it a stable ID |
| **ByteTrack** | The tracker used here; notable for re-using low-confidence detections |
| **Kalman filter** | Predicts an object's next position from its recent motion |
| **Linear assignment** | Optimally pairing detections with tracks (the `lap` library) |
| **Track ID churn** | The tracker losing and re-acquiring one object under new IDs, inflating counts |
| **ROI** | Region of Interest — the polygon we restrict counting to |
| **Occupancy** | Vehicles visible in a single frame |
| **MPS** | Metal Performance Shaders — Apple's GPU backend for PyTorch |
| **BGR** | OpenCV's channel order (blue, green, red) — not RGB |
| **H.264 / mp4v** | Video codecs. Browsers play H.264; most refuse mp4v |

---

## Reproduce every number in this document

```bash
# the imgsz table (Part 3.6) -> prints 9 / 24 / 39 / 66 / 101
for s in 640 960 1280 1600 1920; do
  printf "imgsz %-5s -> " $s
  python detect_vehicles.py --source data/input/parking_lot.jpg --imgsz $s --no-json | grep TOTAL
done

# the confidence table (Part 10)
for c in 0.10 0.15 0.25 0.40 0.60; do
  python detect_vehicles.py --source data/input/parking_lot.jpg --conf $c --no-json | grep TOTAL
done

# the ROI rules (Part 7.4)
for r in bottom center overlap; do
  python detect_vehicles.py --source data/input/parking_lot.jpg \
    --roi data/roi/main_lot.json --roi-rule $r --no-json | grep TOTAL
done

# ID churn (Part 6.4) -> prints 263 / 119 / 75 / 55
for m in 1 3 8 20; do
  printf "min-hits %-2s -> " $m
  python detect_vehicles.py --source data/input/street_traffic.mp4 \
    --min-track-hits $m --no-json 2>/dev/null | grep -E "^  TOTAL" | tail -1
done

# the failure case (Part 11.1)
python tools/fetch_samples.py --all
python detect_vehicles.py --source data/input/topdown_lot.mp4
```

**See also:** [README.md](README.md) for setup and commands, [NOTES.md](NOTES.md) for the
approach/observations/limitations note required by the brief.
