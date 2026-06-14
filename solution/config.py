"""Tunable parameters for the alignment pipeline.

Every constant that affects behaviour lives here with a short rationale, so the
algorithm has no magic numbers buried in the logic. Thresholds expressed as
*percentiles* are deliberate: they adapt to each village's own signal distribution,
which is what lets the same code run unchanged across very different terrain (large
open fields vs. dense small plots).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentConfig:
    # --- search geometry (metres) ---
    coarse_search_m: float = 40.0
    """Half-width of the coarse translation search. Observed drift in the data tops out
    around 20 m; 40 m leaves margin without inviting matches onto distant features."""

    refine_search_m: float = 12.0
    """Half-width of the local refinement search performed around a neighbour-suggested
    position."""

    neighbour_radius_m: float = 600.0
    """Radius within which trusted neighbours vote on the local drift. Expanded by
    2.5x as a fallback when too few neighbours are found nearby."""

    # --- edge map construction ---
    edge_blur_sigma: float = 2.0
    """Gaussian blur applied to the gradient magnitude; widens edges so the match peak
    is smooth rather than knife-edged."""

    edge_clip_quantile: float = 0.35
    """Gradient values are capped at this fraction of their 99th-percentile and
    rescaled. Capping stops bright linear features (roads, canals) from dominating the
    correlation and dragging outlines off the true field edge."""

    imagery_weight: float = 0.5
    """Blend weight for the imagery-derived edge map; the boundary-hint raster gets the
    complement (1 - imagery_weight)."""

    # --- trusted-neighbour selection (percentiles over the village) ---
    trust_peak_pct: int = 55
    trust_min_gain: float = 1.25
    """A parcel is a trusted drift witness only if its match is off-border, its peak
    response is above the village's 55th percentile, and the shift improved the match
    by at least this factor."""

    # --- decision thresholds (percentiles over the village) ---
    good_peak_pct: int = 35
    strong_peak_pct: int = 55
    weak_peak_pct: int = 20
    aligned_zero_pct: int = 90
    """Above the 90th-percentile of baseline (no-shift) support, with negligible gain,
    a parcel is treated as already aligned and left in place."""

    good_min_gain: float = 1.12
    aligned_max_gain: float = 1.08

    # --- neighbour agreement (metres) ---
    agree_tolerance_m: float = 12.0
    """A parcel's own shift is accepted when it lands within this distance of the
    neighbour median (or within the neighbourhood spread, whichever is larger)."""

    override_min_disagreement_m: float = 30.0
    override_max_spread_m: float = 15.0
    rescue_max_spread_m: float = 22.0

    # --- confidence calibration ---
    n_synthetic_trials: int = 250
    """Number of known-offset trials used to learn the confidence mapping empirically."""

    synthetic_max_offset_m: float = 18.0
    synthetic_success_m: float = 5.0
    """A synthetic trial counts as recovered if the estimated shift lands within 5 m of
    the parcel's reference shift."""

    calibration_bins: int = 6
    signal_weights: tuple[float, float, float] = (0.5, 0.2, 0.3)
    """Relative weights for (peak, gain, neighbour-agreement) when ranking match
    quality for the confidence score. Chosen on synthetic trials (best AUC)."""

    random_seed: int = 11
