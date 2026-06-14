"""Unit tests for the decision policy and confidence calibration.

These cover the judgement that matters most: when to trust a parcel's own match, when
to defer to neighbours, when to flag, and that confidence is monotonic in match quality.
All deterministic, no imagery required. Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np

from solution.calibration import fit_calibration
from solution.config import AlignmentConfig
from solution.consensus import NeighbourVote
from solution.decision import ParcelMatch, PeakThresholds, decide

CONFIG = AlignmentConfig()
THRESHOLDS = PeakThresholds(good=0.54, strong=0.57, weak=0.51, aligned_zero=0.51)


def _trivial_calibration():
    """A calibration whose confidence rises with peak, for assertions about ordering."""
    rng = np.random.default_rng(0)
    peaks = rng.uniform(0.4, 0.8, 400)
    signals = np.column_stack([peaks, np.full(400, 1.3), np.full(400, 5.0)])
    successes = peaks > 0.58  # higher peak => more likely recovered
    return fit_calibration(signals, successes, CONFIG)


def _match(**overrides):
    base = dict(shift_dx_m=12.0, shift_dy_m=-4.0, peak=0.62, zero=0.40,
                gain=1.55, at_border=False, usable=True)
    base.update(overrides)
    return ParcelMatch(**base)


def test_strong_match_with_agreeing_neighbour_is_corrected():
    vote = NeighbourVote(median_dx_m=11.0, median_dy_m=-3.0, spread_m=8.0, n_witnesses=20)
    decision = decide(_match(), vote, THRESHOLDS, _trivial_calibration(), CONFIG)
    assert decision.status == "corrected"
    assert decision.reason == "own-match-neighbour-agree"
    assert decision.shift_dx_m == 12.0  # keeps its own estimate


def test_weak_match_without_neighbours_is_flagged():
    weak = _match(peak=0.30, gain=1.02, usable=True)
    decision = decide(weak, None, THRESHOLDS, _trivial_calibration(), CONFIG)
    assert decision.status == "flagged"
    assert decision.confidence == 0.0


def test_already_aligned_parcel_is_left_in_place():
    aligned = _match(shift_dx_m=0.3, shift_dy_m=0.2, peak=0.6, zero=0.6, gain=1.02)
    decision = decide(aligned, None, THRESHOLDS, _trivial_calibration(), CONFIG)
    assert decision.status == "corrected"
    assert decision.reason == "already-aligned"
    assert (decision.shift_dx_m, decision.shift_dy_m) == (0.0, 0.0)


def test_long_jump_against_tight_neighbourhood_requests_override():
    far = _match(shift_dx_m=45.0, shift_dy_m=40.0, peak=0.6, gain=1.5)
    vote = NeighbourVote(median_dx_m=10.0, median_dy_m=-2.0, spread_m=6.0, n_witnesses=25)
    decision = decide(far, vote, THRESHOLDS, _trivial_calibration(), CONFIG)
    assert decision.reason == "neighbour-override"
    assert decision.refine_around == (10.0, -2.0)


def test_weak_match_rescued_by_tight_neighbourhood():
    weak = _match(peak=0.32, gain=1.01, usable=True)
    vote = NeighbourVote(median_dx_m=14.0, median_dy_m=-5.0, spread_m=9.0, n_witnesses=15)
    decision = decide(weak, vote, THRESHOLDS, _trivial_calibration(), CONFIG)
    assert decision.reason == "neighbour-rescue"
    assert decision.refine_around == (14.0, -5.0)


def test_confidence_increases_with_peak():
    calibration = _trivial_calibration()
    low = calibration.confidence(peak=0.45, gain=1.3, disagreement_m=5.0)
    high = calibration.confidence(peak=0.78, gain=1.3, disagreement_m=5.0)
    assert high > low
    assert 0.05 <= low <= 0.95 and 0.05 <= high <= 0.95


def test_calibration_auc_is_meaningful_when_signal_separates():
    calibration = _trivial_calibration()
    assert calibration.auc is not None
    assert calibration.auc > 0.7  # peak cleanly separates success here
