"""Per-parcel decision policy.

Given a parcel's own match and its neighbourhood vote, decide whether to correct it
(and by how much, with what confidence) or to flag it as not confidently placeable.
The policy favours a parcel's own strong match, uses neighbours to veto suspicious
long jumps and to rescue weak matches, and leaves already-aligned parcels untouched.

Everything here works in metres and is free of I/O, so it is straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solution.calibration import CalibrationModel
from solution.config import AlignmentConfig
from solution.consensus import NeighbourVote

_NO_NEIGHBOUR_DISAGREEMENT_M = 30.0  # stand-in distance when a parcel has no witnesses


@dataclass(frozen=True)
class PeakThresholds:
    """Village-relative peak-response cut-offs, derived from percentiles at runtime."""

    good: float
    strong: float
    weak: float
    aligned_zero: float


@dataclass(frozen=True)
class ParcelMatch:
    """A parcel's own match, in metre-space, as the decision policy needs it."""

    shift_dx_m: float
    shift_dy_m: float
    peak: float
    zero: float
    gain: float
    at_border: bool
    usable: bool


@dataclass(frozen=True)
class Decision:
    status: str          # "corrected" or "flagged"
    shift_dx_m: float    # 0.0 for flagged / already-aligned
    shift_dy_m: float
    confidence: float
    reason: str
    refine_around: tuple[float, float] | None = None
    """When set, the pipeline runs a local refinement around this (dx, dy) and confirms
    real edge support before committing. Keeps the I/O-heavy step out of the policy."""


def decide(match: ParcelMatch, vote: NeighbourVote | None,
           thresholds: PeakThresholds, calibration: CalibrationModel,
           config: AlignmentConfig) -> Decision:
    """Map a parcel's evidence to a correction or a flag. Pure function."""
    disagreement = (
        float(np.hypot(match.shift_dx_m - vote.median_dx_m, match.shift_dy_m - vote.median_dy_m))
        if vote else _NO_NEIGHBOUR_DISAGREEMENT_M
    )

    if match.usable and _is_already_aligned(match, vote, thresholds, config):
        return Decision("corrected", 0.0, 0.0, 0.62, "already-aligned")

    own_is_good = (match.usable and match.peak >= thresholds.good
                   and match.gain >= config.good_min_gain)
    if own_is_good:
        return _good_match_decision(match, vote, disagreement, thresholds, calibration, config)

    return _weak_match_decision(match, vote, calibration, config)


def _is_already_aligned(match, vote, thresholds, config) -> bool:
    negligible_vote = vote is None or np.hypot(vote.median_dx_m, vote.median_dy_m) < 10
    return (match.zero >= thresholds.aligned_zero
            and match.gain < config.aligned_max_gain
            and negligible_vote)


def _good_match_decision(match, vote, disagreement, thresholds, calibration, config) -> Decision:
    own = (match.shift_dx_m, match.shift_dy_m)
    if vote is None:
        conf = min(0.55, calibration.confidence(match.peak, match.gain, _NO_NEIGHBOUR_DISAGREEMENT_M))
        return Decision("corrected", *own, conf, "own-match-isolated")

    if disagreement <= max(config.agree_tolerance_m, vote.spread_m):
        conf = calibration.confidence(match.peak, match.gain, disagreement)
        return Decision("corrected", *own, conf, "own-match-neighbour-agree")

    if (disagreement > config.override_min_disagreement_m
            and vote.spread_m <= config.override_max_spread_m):
        conf = min(0.6, calibration.confidence(match.peak, 1.2, 0.0))
        return Decision("corrected", vote.median_dx_m, vote.median_dy_m, conf,
                        "neighbour-override", refine_around=(vote.median_dx_m, vote.median_dy_m))

    conf = calibration.confidence(match.peak, match.gain, disagreement)
    return Decision("corrected", *own, conf, "own-match-uncertain")


def _weak_match_decision(match, vote, calibration, config) -> Decision:
    if vote is not None and vote.spread_m <= config.rescue_max_spread_m:
        conf = min(0.6, calibration.confidence(max(match.peak, 0.0), 1.15, 0.0))
        return Decision("corrected", vote.median_dx_m, vote.median_dy_m, conf,
                        "neighbour-rescue", refine_around=(vote.median_dx_m, vote.median_dy_m))
    return Decision("flagged", 0.0, 0.0, 0.0, "weak-signal")
