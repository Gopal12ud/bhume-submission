"""Neighbourhood drift consensus.

The georeferencing error is locally coherent: nearby parcels drift by similar amounts.
We pick *trusted witnesses* (parcels with strong, unambiguous matches) and let them vote
on the local drift, which lets us sanity-check, override, or rescue weaker matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solution.config import AlignmentConfig


@dataclass(frozen=True)
class NeighbourVote:
    median_dx_m: float
    median_dy_m: float
    spread_m: float
    n_witnesses: int


class DriftConsensus:
    """Spatial index over per-parcel match results for neighbourhood queries.

    All arrays are parallel to the parcel order passed in. Centroids are in the
    imagery CRS (metres), so plain Euclidean distance is valid.
    """

    def __init__(self, centroids_x, centroids_y, shifts_x, shifts_y,
                 is_ok, at_border, gains, peaks, config: AlignmentConfig,
                 strong_peak: float):
        self._cx = np.asarray(centroids_x)
        self._cy = np.asarray(centroids_y)
        self._dx = np.asarray(shifts_x)
        self._dy = np.asarray(shifts_y)
        self._config = config
        self._trusted = (
            np.asarray(is_ok)
            & ~np.asarray(at_border)
            & (np.asarray(gains) >= config.trust_min_gain)
            & (np.asarray(peaks) >= strong_peak)
        )

    @property
    def n_trusted(self) -> int:
        return int(self._trusted.sum())

    def vote(self, index: int) -> NeighbourVote | None:
        """Median drift of trusted witnesses around parcel ``index`` (None if too few)."""
        base_radius = self._config.neighbour_radius_m
        for radius in (base_radius, base_radius * 2.5):
            near = (
                self._trusted
                & (np.abs(self._cx - self._cx[index]) < radius)
                & (np.abs(self._cy - self._cy[index]) < radius)
            )
            near[index] = False
            if near.sum() >= 5:
                mdx = float(np.median(self._dx[near]))
                mdy = float(np.median(self._dy[near]))
                spread = float(np.median(np.hypot(self._dx[near] - mdx, self._dy[near] - mdy)))
                return NeighbourVote(mdx, mdy, spread, int(near.sum()))
        return None
