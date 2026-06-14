"""Image-space matching primitives.

These functions turn a parcel and its surrounding imagery into a translation estimate:
build an edge map that emphasises field boundaries, rasterise the parcel outline, and
slide the outline over the edge map to find the offset of maximum response.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio import features
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter, sobel, zoom
from scipy.signal import fftconvolve

from solution.config import AlignmentConfig


@dataclass(frozen=True)
class MatchResult:
    """Outcome of sliding one outline over its local edge map.

    Offsets are in pixels; convert with the patch transform. ``peak`` is the best
    correlation response normalised by outline length, ``zero`` the response at no
    shift, and ``gain = peak / zero`` measures how much the shift improved the fit.
    ``at_border`` flags a peak pinned to the search edge (an unreliable match).
    """

    dx_px: int
    dy_px: int
    peak: float
    zero: float
    at_border: bool

    @property
    def gain(self) -> float:
        return self.peak / (self.zero + 1e-6)


def build_edge_map(patch, boundary_src, config: AlignmentConfig) -> np.ndarray:
    """Combine an imagery gradient with the optional boundary-hint raster.

    The gradient is capped (see ``edge_clip_quantile``) so no single bright feature
    dominates; the boundary raster, where available, is blended in to reinforce real
    field edges. Returns a float array in roughly [0, 1] the same size as the patch.
    """
    gray = patch.image.mean(axis=2).astype("float32")
    gradient = gaussian_filter(np.hypot(sobel(gray, 1), sobel(gray, 0)), config.edge_blur_sigma)
    cap = config.edge_clip_quantile
    imagery_edges = np.minimum(gradient / (np.percentile(gradient, 99) + 1e-6), cap) / cap

    if boundary_src is None:
        return imagery_edges

    window = from_bounds(*patch.bounds, transform=boundary_src.transform)
    hint = boundary_src.read(1, window=window, boundless=True, fill_value=0).astype("float32") / 255.0
    if hint.size < 4:
        return imagery_edges

    hint = zoom(hint, (imagery_edges.shape[0] / hint.shape[0],
                       imagery_edges.shape[1] / hint.shape[1]), order=1)
    hint = gaussian_filter(hint, 1.5)
    return config.imagery_weight * imagery_edges + (1 - config.imagery_weight) * hint


def rasterise_outline(outline_geom, shape, transform) -> np.ndarray:
    """Rasterise a parcel boundary (in imagery CRS) to a 1-pixel-wide line mask."""
    polygons = outline_geom.geoms if outline_geom.geom_type == "MultiPolygon" else [outline_geom]
    mask = features.rasterize(
        ((poly.boundary, 1) for poly in polygons),
        out_shape=shape, transform=transform, fill=0, all_touched=True,
    )
    return mask.astype("float32")


def match_outline(edge_map: np.ndarray, outline_mask: np.ndarray,
                  pixel_size_m: float, search_m: float) -> MatchResult:
    """Find the translation of ``outline_mask`` that maximises overlap with ``edge_map``.

    Uses an FFT cross-correlation (every candidate shift evaluated at once), then takes
    the best offset within ``search_m`` of centre.
    """
    correlation = fftconvolve(edge_map, outline_mask[::-1, ::-1], mode="same")
    cy, cx = edge_map.shape[0] // 2, edge_map.shape[1] // 2
    radius = max(1, int(search_m / pixel_size_m))

    y0, x0 = max(0, cy - radius), max(0, cx - radius)
    window = correlation[y0:cy + radius + 1, x0:cx + radius + 1]
    peak_y, peak_x = np.unravel_index(np.argmax(window), window.shape)
    dy_px, dx_px = peak_y + y0 - cy, peak_x + x0 - cx

    length = outline_mask.sum() + 1e-6
    at_border = abs(dy_px) >= radius - 1 or abs(dx_px) >= radius - 1
    return MatchResult(
        dx_px=int(dx_px), dy_px=int(dy_px),
        peak=float(window.max() / length),
        zero=float(correlation[cy, cx] / length),
        at_border=bool(at_border),
    )
