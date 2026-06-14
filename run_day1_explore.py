"""
Day 1 — Load a village and look at the data.

Goal for today: confirm the kit works, see how many parcels a village has, and save
one image patch so you can see the satellite imagery your method will work from.

Run:
    python run_day1_explore.py
"""

from PIL import Image

from bhume import load, patch_for_plot
from bhume.geo import open_imagery

VILLAGE_DIR = "data/vadnerbhairav"


def main() -> None:
    village = load(VILLAGE_DIR)
    n_truths = 0 if village.example_truths is None else len(village.example_truths)
    print(f"Loaded village: {village.slug}")
    print(f"  parcels:        {len(village.plots)}")
    print(f"  example truths: {n_truths}")
    print(f"  boundary hints: {'yes' if village.boundaries_path else 'no'}")

    # Save the imagery under the first parcel, so you can see what you are matching against.
    first_parcel = village.plots.index[0]
    with open_imagery(village.imagery_path) as imagery:
        patch = patch_for_plot(imagery, village.plot(first_parcel), pad_m=30)
    Image.fromarray(patch.image).save("day1_patch.png")
    print(f"  saved day1_patch.png  (imagery under parcel {first_parcel}, shape {patch.image.shape})")


if __name__ == "__main__":
    main()
