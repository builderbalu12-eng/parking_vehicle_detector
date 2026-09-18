# Sample outputs

Produced by the commands in the main README, so a reviewer can see results without
running anything.

| File | Command |
|---|---|
| `parking_lot_annotated.jpg` | `python detect_vehicles.py --source data/input/parking_lot.jpg` |
| `parking_lot_roi_annotated.jpg` | the same, plus `--roi data/roi/main_lot.json` |
| `street_traffic_annotated.mp4` | `python detect_vehicles.py --source data/input/street_traffic.mp4` |

The `*_counts.json` files record the settings used alongside the counts, and
`street_traffic_per_frame.csv` has occupancy for every frame.

101 vehicles are found across the whole lot photo; restricting to the `main_lot` ROI
gives 47, excluding the separate right-hand lot and the public street.
