"""Empirical confidence calibration.

Rather than inventing a confidence formula, we measure one. We take parcels with a
clear match, displace them by a known random offset, and check whether the matcher
recovers it. The relationship between match-quality signals and recovery success
becomes the confidence mapping, and its AUC tells us how well confidence tracks
correctness. This is honest self-assessment that needs no ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solution.config import AlignmentConfig


@dataclass
class CalibrationModel:
    """Maps a match-quality score in [0, 1] to a calibrated confidence.

    ``bin_edges`` and ``bin_success_rates`` come from synthetic trials; ``auc`` reports
    how well the score separates recovered from failed trials (0.5 = chance).
    """

    bin_edges: list[float]
    bin_success_rates: list[float]
    auc: float | None
    peak_ref: list[float]
    gain_ref: list[float]
    agreement_ref: list[float]
    weights: tuple[float, float, float]

    def quality_score(self, peak: float, gain: float, disagreement_m: float) -> float:
        """Rank a match by its quality signals, each mapped to its empirical quantile."""
        w_peak, w_gain, w_agree = self.weights
        return (
            w_peak * _quantile_of(peak, self.peak_ref)
            + w_gain * _quantile_of(gain, self.gain_ref)
            + w_agree * _quantile_of(-disagreement_m, self.agreement_ref)
        )

    def confidence(self, peak: float, gain: float, disagreement_m: float) -> float:
        score = self.quality_score(peak, gain, disagreement_m)
        rate = self.bin_success_rates[-1]
        for edge, success in zip(self.bin_edges, self.bin_success_rates):
            if score <= edge:
                rate = success
                break
        # Nudge within a bin so confidence is monotonic in score, then clamp.
        return float(np.clip(rate + 0.06 * (score - 0.5), 0.05, 0.95))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "bin_edges": self.bin_edges,
            "bin_success_rates": self.bin_success_rates,
            "auc": self.auc,
        }, indent=2))


def _quantile_of(value: float, sorted_ref: list[float]) -> float:
    return float(np.searchsorted(sorted_ref, value) / max(1, len(sorted_ref)))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives, negatives = scores[labels], scores[~labels]
    if not len(positives) or not len(negatives):
        return None
    wins = np.mean([(p > negatives).mean() + 0.5 * (p == negatives).mean() for p in positives])
    return round(float(wins), 3)


def fit_calibration(signals, successes, config: AlignmentConfig) -> CalibrationModel:
    """Build a CalibrationModel from synthetic-trial ``signals`` and ``successes``.

    ``signals`` is an (n, 3) array of (peak, gain, disagreement_m); ``successes`` is a
    boolean array of whether each trial's known offset was recovered.
    """
    signals = np.asarray(signals, dtype=float)
    successes = np.asarray(successes, dtype=bool)
    peak_ref = sorted(signals[:, 0])
    gain_ref = sorted(signals[:, 1])
    agreement_ref = sorted(-signals[:, 2])

    model = CalibrationModel(
        bin_edges=[], bin_success_rates=[], auc=None,
        peak_ref=peak_ref, gain_ref=gain_ref, agreement_ref=agreement_ref,
        weights=config.signal_weights,
    )
    scores = np.array([model.quality_score(*row) for row in signals])
    order = np.argsort(scores)
    for chunk in np.array_split(order, config.calibration_bins):
        model.bin_edges.append(float(scores[chunk].max()))
        model.bin_success_rates.append(float(successes[chunk].mean()))
    model.auc = _auc(scores, successes)
    return model
