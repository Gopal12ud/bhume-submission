"""
Day 4 — Run the full pipeline over a whole village.

This is the main deliverable. It runs all three passes — match every parcel, learn a
confidence mapping from synthetic-drift trials, then decide (correct / flag) with
neighbourhood checks — writes predictions.geojson into the village folder, and prints
the self-score against the public example truths.

The village folder is a command-line argument, so the same code runs on either village
with no change:

    python run_day4_full_pipeline.py                  # defaults to Vadnerbhairav
    python run_day4_full_pipeline.py data/malatavadi  # second village, same code
"""

import sys

from bhume import load, score

from solution.config import AlignmentConfig
from solution.pipeline import run_village

DEFAULT_VILLAGE = "data/vadnerbhairav"


def main() -> None:
    village_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VILLAGE
    config = AlignmentConfig()

    predictions = run_village(village_dir, config=config,
                              calibration_out="calibration.json")

    village = load(village_dir)
    if village.example_truths is not None:
        print()
        print(score(predictions, village))


if __name__ == "__main__":
    main()
