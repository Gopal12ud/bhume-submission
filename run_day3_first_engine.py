"""
Day 3 — First correction engine (template matching), on a few parcels.

For each parcel, this builds an edge map, slides the official outline over it to find the
best shift, and draws the result: red dashed = official, green = the engine's correction.
It also prints the shift and a match score, so you can see where the match is strong
(open fields) and where it is weak or wrong (barren land, parallel edges).

This uses the same matching code as the full pipeline, so what you see here is exactly
what runs at scale.

Run:
    python run_day3_first_engine.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from shapely.affinity import translate

from bhume import load, patch_for_plot
from bhume.geo import geom_to_imagery_crs, open_imagery

from solution.config import AlignmentConfig
from solution.matching import build_edge_map, match_outline, rasterise_outline

VILLAGE_DIR = "data/vadnerbhairav"
PARCELS_TO_SHOW = ["605", "603", "440", "552", "2420", "878"]


def main() -> None:
    config = AlignmentConfig()
    village = load(VILLAGE_DIR)
    boundary_src = rasterio.open(village.boundaries_path) if village.boundaries_path else None

    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    with open_imagery(village.imagery_path) as imagery:
        for ax, parcel_id in zip(axes.flat, PARCELS_TO_SHOW):
            outline = village.plot(parcel_id)
            in_pixels = geom_to_imagery_crs(imagery, outline)

            patch = patch_for_plot(imagery, outline, pad_m=config.coarse_search_m + 25)
            edge_map = build_edge_map(patch, boundary_src, config)
            outline_mask = rasterise_outline(in_pixels, edge_map.shape, patch.transform)
            result = match_outline(edge_map, outline_mask, patch.transform.a,
                                   config.coarse_search_m)

            shift_dx_m = result.dx_px * patch.transform.a
            shift_dy_m = result.dy_px * patch.transform.e
            corrected = translate(in_pixels, xoff=shift_dx_m, yoff=shift_dy_m)
            shift_m = (shift_dx_m ** 2 + shift_dy_m ** 2) ** 0.5
            print(f"parcel {parcel_id:>5}: shift {shift_m:5.1f} m | "
                  f"match {result.peak:.3f} (was {result.zero:.3f}, gain x{result.gain:.2f})")

            left, bottom, right, top = patch.bounds
            ax.imshow(patch.image, extent=(left, right, bottom, top))
            for polygon in (in_pixels.geoms if in_pixels.geom_type == "MultiPolygon" else [in_pixels]):
                ax.plot(*polygon.exterior.xy, "r--", lw=2)
            for polygon in (corrected.geoms if corrected.geom_type == "MultiPolygon" else [corrected]):
                ax.plot(*polygon.exterior.xy, color="lime", lw=2)
            ax.set_title(f"parcel {parcel_id} | {shift_m:.0f} m | x{result.gain:.2f}")
            ax.axis("off")

    if boundary_src is not None:
        boundary_src.close()
    plt.tight_layout()
    plt.savefig("day3_corrections.png", dpi=120)
    print("saved day3_corrections.png  (red dashed = official, green = engine's correction)")


if __name__ == "__main__":
    main()
