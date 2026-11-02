"""
Run time series: masking, cleaning (with censored volumes kept in place as NaN rows), GLM-ready
scaling, and the onset / duration -> volume math shared with mvpa-decoding.

Event windows use mvpa-decoding's syntax:
  {"start": {"reference": "onset" | "offset_end", "offset_seconds": s},
   "end":   {"reference": "onset" | "offset_end", "offset_seconds": s}}
"""

import math

import numpy as np
from nilearn import signal
from nilearn.glm.first_level.first_level import mean_scaling
from nilearn.maskers import NiftiMasker

WINDOW_REFERENCES = {"onset", "offset_end"}


# =====================================================
# volume math
# =====================================================

def compute_volume_range(start_time: float, stop_time: float, tr: float, n_frames: int):
    """Return [start_vol, stop_vol) covering start_time for a span of (stop_time - start_time)
    seconds, clipped to n_frames. The volume count comes from the duration rounded to whole TRs
    (not from flooring the start and ceiling the stop, which inflates every window by a volume),
    with at least one volume kept -- the mvpa-decoding convention."""
    start_vol = int(math.floor(start_time / tr))
    n_volumes = max(1, round((stop_time - start_time) / tr))
    stop_vol = min(start_vol + n_volumes, n_frames)
    return start_vol, max(stop_vol, start_vol)


def event_volumes(onset: float, duration: float, tr: float, n_frames: int, lag: float = 0.0) -> np.ndarray:
    """Volumes spanned by an event shifted by `lag` seconds (e.g. the hemodynamic lag)."""
    lo, hi = compute_volume_range(onset + lag, onset + lag + duration, tr, n_frames)
    return np.arange(lo, hi)


def validate_window(window, path="window") -> list:
    if not isinstance(window, dict):
        return [f"{path}: must be an object with 'start' and 'end'"]
    errors = []
    for key in ("start", "end"):
        bound = window.get(key)
        if not isinstance(bound, dict):
            errors.append(f"{path}.{key}: must be an object")
            continue
        if bound.get("reference") not in WINDOW_REFERENCES:
            errors.append(f"{path}.{key}.reference must be one of {sorted(WINDOW_REFERENCES)}")
        if not isinstance(bound.get("offset_seconds", 0), (int, float)):
            errors.append(f"{path}.{key}.offset_seconds must be a number")
    return errors


def resolve_window_times(window: dict, onset: float, duration: float):
    """-> (start_s, end_s) for one event."""
    def bound(b):
        base = onset if b["reference"] == "onset" else onset + duration
        return base + float(b.get("offset_seconds", 0))
    return bound(window["start"]), bound(window["end"])


def window_volumes(window: dict, onset: float, duration: float, tr: float, n_frames: int) -> np.ndarray:
    start, end = resolve_window_times(window, onset, duration)
    lo, hi = compute_volume_range(start, end, tr, n_frames)
    return np.arange(lo, hi)


# =====================================================
# masking and cleaning
# =====================================================

def load_masked(boldfile, mask_img, tr: float = None, smoothing_fwhm=None):
    """-> (fitted NiftiMasker, raw volumes x voxels array)."""
    masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=smoothing_fwhm, t_r=tr).fit()
    return masker, masker.transform(boldfile)


def clean_run(raw: np.ndarray, tr: float, nuisance=None, detrend=True, standardize="zscore_sample",
              high_pass=None, low_pass=None, regress_confounds=True) -> np.ndarray:
    """nilearn signal.clean of one run (confound regression, detrend, band-pass, z-scoring) with
    censored volumes excluded from the fit and returned as NaN rows, so volume indices keep their
    timing. `nuisance` is a RunNuisance (or None)."""
    n_frames = raw.shape[0]
    keep = np.arange(n_frames) if nuisance is None else nuisance.kept(n_frames)
    conf = None
    if regress_confounds and nuisance is not None and nuisance.confounds is not None:
        conf = nuisance.confounds.values
    cleaned = signal.clean(raw, detrend=detrend, standardize=standardize, high_pass=high_pass, low_pass=low_pass,
                           t_r=tr, confounds=conf, sample_mask=None if len(keep) == n_frames else keep)
    out = np.full(raw.shape, np.nan)
    out[keep] = cleaned
    return out


def glm_ready(raw: np.ndarray, nuisance=None, signal_scaling=0) -> np.ndarray:
    """Kept volumes only, mean-scaled as nilearn's FirstLevelModel does (signal_scaling 0 =
    percent signal change per voxel, 1 = per volume, False = none)."""
    keep = np.arange(raw.shape[0]) if nuisance is None else nuisance.kept(raw.shape[0])
    data = raw[keep]
    if signal_scaling is not False:
        data, _ = mean_scaling(data, axis=signal_scaling)
    return data
