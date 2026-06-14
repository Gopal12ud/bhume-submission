"""End-to-end village pipeline: match every parcel, calibrate confidence, decide.

Three passes:
  1. Match each official outline against its local edge map (coarse search).
  2. Learn the confidence mapping from synthetic known-offset trials.
  3. Decide per parcel (correct / flag), refining neighbour-suggested positions.

Geospatial plumbing (loading, CRS handling, patch extraction, scoring, writing) is
delegated to the provided ``bhume`` starter-kit helpers; this module owns the method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.affinity import translate
from shapely.ops import transform as shapely_transform

from bhume import load, patch_for_plot, write_predictions
from bhume.geo import geom_to_imagery_crs, open_imagery

from solution.calibration import CalibrationModel, fit_calibration
from solution.config import AlignmentConfig
from solution.consensus import DriftConsensus
from solution.decision import Decision, ParcelMatch, PeakThresholds, decide
from solution.matching import build_edge_map, match_outline, rasterise_outline


@dataclass
class _RawMatch:
    """Internal per-parcel match record carried between passes (imagery CRS)."""

    centroid_x: float
    centroid_y: float
    ok: bool = False
    shift_dx_m: float = 0.0
    shift_dy_m: float = 0.0
    peak: float = 0.0
    zero: float = 0.0
    gain: float = 0.0
    at_border: bool = True


def run_village(village_dir: str | Path, config: AlignmentConfig | None = None,
                calibration_out: str | Path | None = None,
                log=print) -> gpd.GeoDataFrame:
    """Run the full pipeline and write ``predictions.geojson`` into the village folder."""
    config = config or AlignmentConfig()
    village = load(str(village_dir))
    parcels = village.plots
    parcel_ids = list(parcels.index)
    boundary_src = (rasterio.open(village.boundaries_path)
                    if village.boundaries_path else None)
    rng = np.random.default_rng(config.random_seed)

    log(f"{village.slug}: {len(parcel_ids)} parcels")
    with open_imagery(village.imagery_path) as imagery:
        matches = _match_all_parcels(imagery, boundary_src, parcels, parcel_ids, config, log)
        thresholds, consensus = _derive_thresholds_and_consensus(matches, parcel_ids, config)
        log(f"  trusted drift witnesses: {consensus.n_trusted}/{len(parcel_ids)}")

        calibration = _calibrate_confidence(
            imagery, boundary_src, parcels, parcel_ids, matches, consensus, config, rng, log)
        log(f"  calibration AUC: {calibration.auc}  "
            f"(bin success rates: {[round(r, 2) for r in calibration.bin_success_rates]})")

        predictions = _decide_all_parcels(
            imagery, boundary_src, parcels, parcel_ids, matches,
            consensus, thresholds, calibration, config, log)

    output_path = Path(village_dir) / "predictions.geojson"
    write_predictions(output_path, predictions)
    log(f"  wrote {output_path}")

    if calibration_out:
        calibration.to_json(calibration_out)
    return predictions


# --- pass 1 -----------------------------------------------------------------

def _match_all_parcels(imagery, boundary_src, parcels, parcel_ids, config, log) -> dict[str, _RawMatch]:
    log("  pass 1/3: matching each parcel...")
    matches: dict[str, _RawMatch] = {}
    started = time.time()
    for i, parcel_id in enumerate(parcel_ids, 1):
        if i % 400 == 0:
            log(f"    {i}/{len(parcel_ids)} ({time.time() - started:.0f}s)")
        geom = parcels.loc[parcel_id, "geometry"]
        outline = geom_to_imagery_crs(imagery, geom)
        centroid = outline.centroid
        record = _RawMatch(centroid_x=centroid.x, centroid_y=centroid.y)
        try:
            result = _match_single(imagery, boundary_src, geom, outline,
                                   config.coarse_search_m, config)
            transform_a, transform_e = _patch_pixel_size(imagery, geom, config)
            record.ok = True
            record.at_border = result.at_border
            record.peak, record.zero, record.gain = result.peak, result.zero, result.gain
            record.shift_dx_m = result.dx_px * transform_a
            record.shift_dy_m = result.dy_px * transform_e
        except (ValueError, IndexError):
            pass  # leave record.ok = False; treated as un-matchable downstream
        matches[parcel_id] = record
    return matches


def _match_single(imagery, boundary_src, geom, outline, search_m, config):
    patch = patch_for_plot(imagery, geom, pad_m=search_m + 25)
    edge_map = build_edge_map(patch, boundary_src, config)
    outline_mask = rasterise_outline(outline, edge_map.shape, patch.transform)
    if outline_mask.sum() < 10:
        raise ValueError("outline too small to rasterise")
    return match_outline(edge_map, outline_mask, patch.transform.a, search_m)


def _patch_pixel_size(imagery, geom, config):
    patch = patch_for_plot(imagery, geom, pad_m=config.coarse_search_m + 25)
    return patch.transform.a, patch.transform.e  # e is negative (row down -> y down)


# --- thresholds + consensus -------------------------------------------------

def _derive_thresholds_and_consensus(matches, parcel_ids, config):
    ok = np.array([matches[p].ok for p in parcel_ids])
    peak = np.array([matches[p].peak for p in parcel_ids])
    zero = np.array([matches[p].zero for p in parcel_ids])

    thresholds = PeakThresholds(
        good=float(np.percentile(peak[ok], config.good_peak_pct)),
        strong=float(np.percentile(peak[ok], config.strong_peak_pct)),
        weak=float(np.percentile(peak[ok], config.weak_peak_pct)),
        aligned_zero=float(np.percentile(zero[ok], config.aligned_zero_pct)),
    )
    consensus = DriftConsensus(
        centroids_x=[matches[p].centroid_x for p in parcel_ids],
        centroids_y=[matches[p].centroid_y for p in parcel_ids],
        shifts_x=[matches[p].shift_dx_m for p in parcel_ids],
        shifts_y=[matches[p].shift_dy_m for p in parcel_ids],
        is_ok=ok, at_border=[matches[p].at_border for p in parcel_ids],
        gains=[matches[p].gain for p in parcel_ids], peaks=peak,
        config=config, strong_peak=thresholds.strong,
    )
    return thresholds, consensus


# --- pass 2: calibration ----------------------------------------------------

def _calibrate_confidence(imagery, boundary_src, parcels, parcel_ids, matches,
                          consensus, config, rng, log) -> CalibrationModel:
    log(f"  pass 2/3: {config.n_synthetic_trials} synthetic recovery trials...")
    eligible = [i for i, p in enumerate(parcel_ids)
                if matches[p].ok and not matches[p].at_border
                and np.hypot(matches[p].shift_dx_m, matches[p].shift_dy_m) <= 30]
    sample = rng.choice(eligible, min(config.n_synthetic_trials, len(eligible)), replace=False)

    signals, successes = [], []
    for index in sample:
        parcel_id = parcel_ids[index]
        geom = parcels.loc[parcel_id, "geometry"]
        try:
            outline = geom_to_imagery_crs(imagery, geom)
            patch = patch_for_plot(imagery, geom, pad_m=80)
            edge_map = build_edge_map(patch, boundary_src, config)
            offset = rng.uniform(-config.synthetic_max_offset_m, config.synthetic_max_offset_m, 2)
            displaced = translate(outline, offset[0], offset[1])
            mask = rasterise_outline(displaced, edge_map.shape, patch.transform)
            if mask.sum() < 10:
                continue
            result = match_outline(edge_map, mask, patch.transform.a, config.coarse_search_m + 15)
            recovered = (offset[0] + result.dx_px * patch.transform.a,
                         offset[1] + result.dy_px * patch.transform.e)
            error = np.hypot(recovered[0] - matches[parcel_id].shift_dx_m,
                             recovered[1] - matches[parcel_id].shift_dy_m)
            vote = consensus.vote(index)
            disagreement = (float(np.hypot(matches[parcel_id].shift_dx_m - vote.median_dx_m,
                                           matches[parcel_id].shift_dy_m - vote.median_dy_m))
                            if vote else 30.0)
            signals.append((matches[parcel_id].peak, matches[parcel_id].gain, disagreement))
            successes.append(bool(error <= config.synthetic_success_m))
        except (ValueError, IndexError):
            continue

    return fit_calibration(signals, successes, config)


# --- pass 3: decisions ------------------------------------------------------

def _decide_all_parcels(imagery, boundary_src, parcels, parcel_ids, matches,
                        consensus, thresholds, calibration, config, log) -> gpd.GeoDataFrame:
    log("  pass 3/3: deciding + refining...")
    to_lonlat = Transformer.from_crs(imagery.crs, "EPSG:4326", always_xy=True)
    to_lonlat_fn = lambda xs, ys, z=None: to_lonlat.transform(xs, ys)

    rows = []
    for k, parcel_id in enumerate(parcel_ids):
        if (k + 1) % 500 == 0:
            log(f"    {k + 1}/{len(parcel_ids)}")
        geom = parcels.loc[parcel_id, "geometry"]
        outline = geom_to_imagery_crs(imagery, geom)
        raw = matches[parcel_id]
        vote = consensus.vote(k)

        parcel_match = ParcelMatch(
            shift_dx_m=raw.shift_dx_m, shift_dy_m=raw.shift_dy_m,
            peak=raw.peak, zero=raw.zero, gain=raw.gain,
            at_border=raw.at_border, usable=raw.ok and not raw.at_border,
        )
        decision = decide(parcel_match, vote, thresholds, calibration, config)
        decision = _confirm_refinement(imagery, boundary_src, geom, outline,
                                       decision, raw, calibration, config)
        rows.append(_to_feature(parcel_id, geom, outline, decision, to_lonlat_fn))

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _confirm_refinement(imagery, boundary_src, geom, outline, decision: Decision,
                        raw: _RawMatch, calibration, config) -> Decision:
    """For neighbour-suggested positions, refine locally and confirm edge support.

    A neighbour-override is only kept if the refined position matches the imagery at
    least as well as the parcel's own match; otherwise we fall back to the own match.
    A neighbour-rescue is kept only if the refined position has real support.
    """
    if decision.refine_around is None:
        return decision
    start_dx, start_dy = decision.refine_around
    try:
        refined_dx, refined_dy, support = _refine_local(
            imagery, boundary_src, geom, outline, start_dx, start_dy, config)
    except (ValueError, IndexError):
        return decision

    if decision.reason == "neighbour-override":
        if support > raw.peak * 0.98:
            return Decision("corrected", refined_dx, refined_dy, decision.confidence,
                            "neighbour-override")
        conf = min(0.5, calibration.confidence(raw.peak, raw.gain, config.override_min_disagreement_m))
        return Decision("corrected", raw.shift_dx_m, raw.shift_dy_m, conf, "own-vs-neighbour")

    # neighbour-rescue
    if support >= calibration_weak_floor(calibration, config):
        return Decision("corrected", refined_dx, refined_dy, decision.confidence, "neighbour-rescue")
    return Decision("flagged", 0.0, 0.0, 0.0, "weak-signal")


def calibration_weak_floor(calibration, config) -> float:
    """Minimum support a rescued match must show; mirrors the weak-peak percentile."""
    # peak_ref is sorted ascending; the weak percentile indexes into it.
    idx = int(len(calibration.peak_ref) * config.weak_peak_pct / 100)
    return calibration.peak_ref[min(idx, len(calibration.peak_ref) - 1)]


def _refine_local(imagery, boundary_src, geom, outline, start_dx, start_dy, config):
    pad = float(np.hypot(start_dx, start_dy) + config.refine_search_m + 30)
    patch = patch_for_plot(imagery, geom, pad_m=pad)
    edge_map = build_edge_map(patch, boundary_src, config)
    mask = rasterise_outline(translate(outline, start_dx, start_dy), edge_map.shape, patch.transform)
    if mask.sum() < 10:
        raise ValueError("outline too small to rasterise")
    result = match_outline(edge_map, mask, patch.transform.a, config.refine_search_m)
    return (start_dx + result.dx_px * patch.transform.a,
            start_dy + result.dy_px * patch.transform.e,
            result.peak)


def _to_feature(parcel_id, geom, outline, decision: Decision, to_lonlat_fn) -> dict:
    if decision.status == "flagged":
        geometry = geom
    elif abs(decision.shift_dx_m) < 0.5 and abs(decision.shift_dy_m) < 0.5:
        geometry = geom
    else:
        shifted = translate(outline, decision.shift_dx_m, decision.shift_dy_m)
        geometry = shapely_transform(to_lonlat_fn, shifted)

    properties = {
        "plot_number": parcel_id,
        "status": decision.status,
        "method_note": decision.reason,
        "geometry": geometry,
    }
    if decision.status == "corrected":
        properties["confidence"] = round(float(decision.confidence), 2)
    else:
        properties["confidence"] = 0.0
    return properties
