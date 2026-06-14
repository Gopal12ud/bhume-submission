"""
Day 2 — See the drift with your own eyes.

Draws the official parcel outline (red, dashed) over the satellite imagery for a handful
of parcels and saves a grid image. Open fields show the outline sitting just off the
real field; barren parcels show there are no edges to match against.

Run:
    python run_day2_see_drift.py
"""

import matplotlib
matplotlib.use("Agg")  # render straight to a file, no window
import matplotlib.pyplot as plt

from bhume import load, patch_for_plot
from bhume.geo import geom_to_imagery_crs, open_imagery

VILLAGE_DIR = "data/vadnerbhairav"
PARCELS_TO_SHOW = ["605", "603", "440", "552", "2420", "878"]


def main() -> None:
    village = load(VILLAGE_DIR)
    print(f"{village.slug}: drawing {len(PARCELS_TO_SHOW)} parcels")

    with open_imagery(village.imagery_path) as imagery:
        _, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, parcel_id in zip(axes.flat, PARCELS_TO_SHOW):
            try:
                outline = village.plot(parcel_id)
            except KeyError:
                ax.set_title(f"parcel {parcel_id} not found")
                ax.axis("off")
                continue

            patch = patch_for_plot(imagery, outline, pad_m=60)
            left, bottom, right, top = patch.bounds
            ax.imshow(patch.image, extent=(left, right, bottom, top))

            in_pixels = geom_to_imagery_crs(imagery, outline)
            polygons = in_pixels.geoms if in_pixels.geom_type == "MultiPolygon" else [in_pixels]
            for polygon in polygons:
                ax.plot(*polygon.exterior.xy, color="red", linewidth=2, linestyle="--")
            ax.set_title(f"parcel {parcel_id}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("day2_drift.png", dpi=120)
    print("saved day2_drift.png  (red dashed = official outline; note how it sits off the field)")


if __name__ == "__main__":
    main()
