"""parcel_align — align shifted cadastral parcels to satellite imagery.

A small, well-organised solution for the BhuMe boundary take-home. The official parcel
outlines in the land records sit several metres off the real fields; this package finds,
per parcel, the translation that best lands the outline on the field edges visible in
imagery, assigns a calibrated confidence, and flags parcels it cannot place.

Modules:
    config      — all tunable parameters in one documented place
    matching    — edge map, outline rasterisation, FFT template matching
    consensus   — trusted-witness selection and neighbourhood drift voting
    calibration — synthetic-drift confidence model
    decision    — pure correct/flag policy (no I/O, fully testable)
    pipeline    — orchestrates the three passes over a whole village
"""

from solution.config import AlignmentConfig

__all__ = ["AlignmentConfig"]
