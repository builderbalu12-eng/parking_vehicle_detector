# Approach, observations and limitations

All numbers below were measured on this project's samples with `yolo11n.pt`, on an
Apple M-series Mac using the `mps` backend. They are reproducible with the commands
shown.

---

## 1. Approach

**Use the model as-is.** The brief rules out training, and COCO — the dataset YOLO11 is
pre-trained on — already contains `car` (id 2), `motorcycle` (3), `bus` (5) and
`truck` (7). So the work is not modelling; it is everything around the model: filtering
to the right classes, deciding what "a count" means, and presenting the result honestly.

**Filter classes inside the model call, not afterwards.** `detector.py` passes
`classes=[2, 3, 5, 7]` into `model.predict()`. Ultralytics then discards non-vehicle
classes *before* non-maximum suppression. Filtering afterwards would let a pedestrian
standing beside a car suppress that car's box and disappear it from the count.

**Convert once, at the boundary.** `detector.py` is the only file that touches raw
Ultralytics objects. It turns them into a plain `Detection` dataclass, so drawing,
counting and ROI logic stay independent of the library's API.

**Match inference resolution to the source.** This turned out to be the single most
important decision in the project — see observation 3.1 below.

**Two different counts for video.** Occupancy (vehicles in *this* frame) and unique
vehicles (distinct tracker IDs across the clip) answer different questions. Summing
per-frame counts would be meaningless: a car parked for 300 frames would be counted
300 times.

**Anchor the ROI test at the bottom of the box.** In an angled lot view a box's
centroid often floats over the *neighbouring* bay, while the bottom-centre — roughly
where the tyres meet the tarmac — sits in the correct one. `--roi-rule` exposes
`center` and `overlap` for top-down views and boundary-straddling vehicles.

---

## 2. Reading YOLO's output

A single inference returns a `Results` object whose `.boxes` holds parallel tensors,
one row per surviving detection:

| Attribute | Shape | Meaning |
|---|---|---|
| `boxes.xyxy` | (N, 4) | `[x1, y1, x2, y2]` — top-left and bottom-right corners, **in pixels of the input frame** |
| `boxes.conf` | (N,) | confidence in `[0, 1]` for the assigned class |
| `boxes.cls` | (N,) | COCO class id as a float, cast to int |
| `boxes.id` | (N,) | tracker ID, present only when `model.track()` was used, otherwise `None` |

`detector.py::_to_detections` is the ~15 lines that convert this into `Detection`
objects.

**What the confidence score actually is.** It is the model's score for that box being
that class, *after* the confidence threshold and NMS have already been applied. It is
**not a calibrated probability** — a `0.90` car is not "90% likely to be a car" in any
frequentist sense. It is a ranking signal, useful for comparing detections within one
image and for thresholding, not for reasoning about real-world likelihood.

**What `--iou` does.** The model proposes many overlapping boxes for the same object.
NMS keeps the highest-confidence box and deletes any other box overlapping it by more
than the IoU threshold. Raise it and near-duplicate boxes survive (double counting);
lower it and genuinely adjacent vehicles get merged into one (under counting). In a
packed lot, cars nearly touch, so this threshold is a real trade-off rather than a
formality.

**`xyxy` vs `xywh`.** Ultralytics exposes both. This project uses `xyxy` because
OpenCV's `rectangle()` takes two corners, so no conversion is needed.

---

## 3. Observations

### 3.1 Inference resolution mattered more than anything else

YOLO letterboxes every input to a square of `imgsz` pixels. The 2048×932 sample lot
squeezed into the conventional 640 px turns a parked car into a ~15 px blob the model
cannot resolve. Same image, same weights, same threshold — only `--imgsz` changes:

| `--imgsz` | vehicles found | inference time |
|---:|---:|---:|
| 640 | 9 | 0.02 s |
| 960 | 24 | 0.03 s |
| 1280 | 39 | 0.02 s |
| 1600 | 66 | 0.03 s |
| **1920** | **101** | 0.04 s |
| 2048 | 103 | 0.04 s |

**9 → 101 detections for +0.02 s.** Because the cost is nearly free at this scale and
the accuracy difference is enormous, `--imgsz` defaults to `auto`: the frame's long
side, snapped to a multiple of 32 and capped at 1920. The cap is deliberate — the gain
from 1920 to 2048 is 2 detections.

*Reproduce:* `python detect_vehicles.py --source data/input/parking_lot.jpg --imgsz 640`

### 3.2 The confidence threshold trades recall against precision

| `--conf` | total | car | truck | bus |
|---:|---:|---:|---:|---:|
| 0.10 | 135 | 122 | 12 | 1 |
| 0.15 | 123 | 115 | 7 | 1 |
| **0.25** | **101** | 96 | 4 | 1 |
| 0.40 | 72 | 70 | 2 | 0 |
| 0.60 | 34 | 34 | 0 | 0 |

The default 0.25 is a compromise. At 0.10 the extra detections are increasingly
duplicates and shadows; at 0.60 more than half the genuinely visible cars are gone.
There is no threshold at which the count is simply "correct".

### 3.3 Tracker ID churn inflates the unique count

On the 17-second street clip the tracker emitted **263** distinct IDs, yet the busiest
single frame held only 15 vehicles. The camera is moving, so ByteTrack repeatedly loses
a vehicle and re-acquires it under a fresh ID.

| track must survive | unique vehicles kept |
|---:|---:|
| 1 frame | 263 |
| 2 frames | 263 |
| **3 frames** | **119** |
| 8 frames | 75 |
| 20 frames | 55 |

The median track lasted **2 frames** — 144 IDs existed for exactly two frames and then
vanished. Requiring 3 frames (`--min-track-hits`, the default) removes those ghosts and
cuts the count from 263 to 119. Both numbers are reported, because the gap between them
is itself the honest measure of how much to trust the figure.

### 3.4 Speed

| Device | Per 1920 px image | Video (1226×370) |
|---|---:|---:|
| `mps` (Apple GPU) | 41 ms | 41 FPS |
| `cpu` | 200 ms | — |

MPS is roughly **5× faster** than CPU here. Video processing runs faster than real time,
so a live camera feed would be feasible.

---

## 4. Limitations

### 4.1 Top-down views break it — a bundled demonstration

COCO photographs are overwhelmingly taken from ground level. A camera pointed straight
down sees a car shape the model has essentially never been trained on. `topdown_lot.mp4`
(fetch with `python tools/fetch_samples.py --all`) is included **specifically to show
this failure**:

| Setting | Frames with any detection | Detections |
|---|---:|---:|
| defaults | 28 of 377 | 27 car, 2 bus, 1 truck |
| `--conf 0.10 --imgsz 1920` | 69 of 377 | 51 car, 12 truck, **35 bus** |

Pushing the threshold down does not recover the vehicles; it mostly invents **35 buses**
in a lot containing none. This is the clearest limitation of the whole solution: it is a
*data* mismatch, and no amount of parameter tuning fixes it. The real fix is a model
fine-tuned on overhead imagery (e.g. DOTA/VisDrone-style datasets), which the brief
excludes.

The elevated-but-oblique sample lot works well precisely because it still resembles the
viewpoint COCO was shot from.

### 4.2 Class confusion distorts the per-class breakdown

`car` vs `truck` is the weak boundary. SUVs, vans and pickups sit between the two
COCO definitions and flip depending on angle, colour and occlusion. In the sweep above,
`truck` moves from 12 to 2 purely by changing the confidence threshold, while the
vehicles in the image obviously did not change. `bus` vs `truck` confuses similarly on
large panel vans.

**The total is considerably more trustworthy than the per-class split.** Treat per-class
numbers as indicative.

### 4.3 Occlusion in dense lots

Tightly packed rows mean vehicles partially hide one another. Two effects follow: a
half-hidden car may fall below the confidence threshold, and NMS may merge two adjacent
cars into one box. Both push the count *down*, so in a full lot this tool is more likely
to under-count than over-count.

### 4.4 Small and distant vehicles

Even at `imgsz 1920`, vehicles at the far edge of the lot are only a few pixels tall and
are missed. Tiling the image (e.g. SAHI-style sliced inference) would help; it is not
implemented here.

### 4.5 The ROI is image-space only

An ROI is a polygon in pixel coordinates, stored normalised so it survives a change of
resolution. It does **not** survive the camera moving, panning or zooming — the region
would then cover the wrong ground. There is no homography or camera calibration, so the
ROI cannot be expressed in real-world coordinates.

### 4.6 No empty-bay awareness

This counts **vehicles**, not **spaces**. It cannot report free bays without a map of
where the bays are; occupancy here means "how many vehicles are visible", not "the lot
is 80% full".

### 4.7 Other

- **Stationary vehicles and tracking.** A parked car that is occluded for a while may be
  re-acquired as a new ID, so `--min-track-hits` mitigates but does not eliminate churn.
- **Single model size.** Everything is measured with `yolo11n`, the smallest model.
  `yolo11s/m` would detect more, more slowly; `--model` switches it.
- **Night, rain, glare** are untested — all samples are daylight.
- **No tests.** For a deliverable of this size the verification was manual and visual;
  a production version would want unit tests on the ROI geometry and the counting logic.

---

## 5. If this were taken further

1. Fine-tune on overhead parking imagery — the fix for limitation 4.1, and the highest
   value change by a wide margin.
2. Sliced inference (SAHI) for the small-vehicle problem in 4.4.
3. Map the actual parking bays, so the output becomes "17 of 60 bays free" rather than a
   raw vehicle count.
4. Calibrate confidence against a hand-labelled subset, so the threshold is chosen from
   data rather than by eye.
