# parcel-align — BhuMe boundary take-home

Align shifted cadastral parcel outlines to satellite imagery, attach a **calibrated
confidence** to each correction, and **flag** the parcels that cannot be placed with
confidence. One method, run unchanged across two very different villages.

| | Median IoU (corrected) | vs official | Accurate @ IoU>=0.5 | Calibration AUC* |
|---|---:|---:|---:|---:|
| **Vadnerbhairav** (large open fields) | **0.87** | 0.61 | 100% | 0.73 |
| **Malatavadi** (dense small plots, *same code*) | **0.78** | 0.51 | -- | 1.00 |

<sub>* AUC is from synthetic-recovery trials (hundreds of self-generated labelled examples), not the handful of public truths.</sub>

---

## The problem

Maharashtra's 7/12 land records were drawn on paper decades ago and georeferenced onto
satellite imagery imperfectly, so a parcel's official outline often sits several metres
from the field it describes -- the **shape is right, the position is wrong**. For each
parcel this tool estimates the translation that lands the outline on the real field
edges, reports a confidence, and flags parcels where the evidence is too weak (barren
land, or dense plots with many ambiguous parallel edges). Correcting every parcel is not
the goal; an honest answer where possible and an honest flag where not is the point.

## How it works (three passes)

1. **Match** -- build an edge map from the imagery gradient (bright roads capped so they
   cannot dominate) blended with the optional boundary-hint raster, then slide the
   official outline over it (FFT cross-correlation) to find the best offset.
2. **Calibrate** -- displace well-matched parcels by a *known* offset and see if the
   matcher recovers it; the link between match quality and recovery success becomes the
   confidence mapping (no ground truth needed).
3. **Decide** -- trust a parcel's own strong match, use neighbours to veto long jumps and
   rescue weak matches, leave already-aligned parcels alone, and flag the rest.

Every threshold is a **percentile of the current village's own** match distribution,
which is what lets the identical code generalise across very different terrain.

## Folder layout

```
.
├── solution/              # the method (clean, reusable modules)
│   ├── config.py          # all tunable parameters, documented
│   ├── matching.py        # edge map + outline rasterise + FFT template match
│   ├── consensus.py       # trusted witnesses + neighbourhood drift voting
│   ├── calibration.py     # synthetic-drift confidence model
│   ├── decision.py        # pure correct/flag policy (testable, no I/O)
│   └── pipeline.py        # runs the three passes over a village
├── bhume/                 # starter-kit helpers (loading, CRS, scoring) -- unchanged
├── tests/test_decision.py # deterministic tests for the decision logic
├── data/                  # village bundles go here (see below)
├── run_day1_explore.py        # load a village, save one imagery patch
├── run_day2_see_drift.py      # draw official outlines over imagery
├── run_day3_first_engine.py   # template match a few parcels, before/after
├── run_day4_full_pipeline.py  # the full deliverable: writes predictions.geojson
└── README.md
```

## Setup

Python 3.12+. Install the dependencies once:

```bash
pip install geopandas rasterio shapely numpy scipy pillow matplotlib
```

Download each village bundle from the BhuMe **Get started** page and unzip under `data/`:

```
data/vadnerbhairav/   input.geojson  imagery.tif  boundaries.tif  example_truths.geojson
data/malatavadi/      (the same four files)
```

## Run it (no install step -- just run from this folder)

```bash
python run_day1_explore.py            # confirm the kit works, see the data
python run_day2_see_drift.py          # see the drift (saves day2_drift.png)
python run_day3_first_engine.py       # first matcher, before/after (day3_corrections.png)

python run_day4_full_pipeline.py                  # full run on Vadnerbhairav
python run_day4_full_pipeline.py data/malatavadi  # same code, second village
```

`run_day4_full_pipeline.py` writes `data/<village>/predictions.geojson` and prints the
self-score against that village's public example truths.

## Tests

```bash
python -m pytest -q
```

Covers the decision policy (own-match acceptance, neighbour veto/rescue, already-aligned
restraint, flagging) and that confidence rises with match quality. Deterministic; no
imagery needed.

## Limitations and next steps

- **Translation only** -- corrects position, not rotation or local stretch.
- **Dense terrain** -- a fixed search radius can lock onto an adjacent field; scaling the
  search to parcel size would help.
- **Thin-hint regions** -- where neither imagery edges nor the boundary raster inform, the
  method correctly flags rather than guesses.

---

<sub>Built with Python, numpy, scipy, rasterio, shapely and geopandas. AI was used to write code
and analyse results. the direction, the diagnosis of an early parallel-edge mismatch, and the
synthetic-calibration design were the my contribution.</sub>
