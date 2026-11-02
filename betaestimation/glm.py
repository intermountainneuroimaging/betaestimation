"""
GLM building blocks (nilearn): single-run first-level fits, least-squares-separate (LSS)
single-trial fits, a concatenated multi-run GLM with shared regressors, contrasts, and mini-block
merging of events.

GLM parameters are one dict ("glm"): any nilearn FirstLevelModel keyword (hrf_model,
drift_model, high_pass, drift_order, fir_delays, min_onset, noise_model, smoothing_fwhm,
signal_scaling, slice_time_ref, minimize_memory, ...) plus these, which this package reads:
  output_type   "t" | "z" | "beta"         which map a contrast returns
  trial_model   "lsa" | "lss"              single-trial estimation (see estimate.py)
  lss_other     "condition" | "single"     how LSS models the other events
"""

import re

import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix, run_glm

OUTPUT_TYPES = {"t": "stat", "z": "z_score", "beta": "effect_size"}
TRIAL_MODELS = {"lsa", "lss"}
LSS_OTHER = {"condition", "single"}
# keys of the glm dict that are this package's, not FirstLevelModel keywords
PACKAGE_KEYS = ("output_type", "trial_model", "lss_other", "confounds")
# keys forwarded to make_first_level_design_matrix when building designs by hand
DESIGN_KEYS = ("hrf_model", "drift_model", "high_pass", "drift_order", "fir_delays", "min_onset")


def safe_name(name) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(name))


def first_level_kwargs(glm: dict) -> dict:
    return {k: v for k, v in (glm or {}).items() if k not in PACKAGE_KEYS}


def design_kwargs(glm: dict) -> dict:
    return {k: glm[k] for k in DESIGN_KEYS if k in (glm or {})}


def frame_times(n_frames: int, tr: float, slice_time_ref: float = 0.0) -> np.ndarray:
    return (np.arange(n_frames) + slice_time_ref) * tr


def merge_blocks(events: pd.DataFrame, gap: float) -> pd.DataFrame:
    """Merge consecutive same-trial_type events separated by <= gap seconds into one boxcar
    (e.g. a localizer mini-block of 3 same-subcategory trials modeled as one regressor event)."""
    events = events.sort_values("onset").reset_index(drop=True)
    rows = []
    for _, ev in events.iterrows():
        if rows and rows[-1]["trial_type"] == ev["trial_type"] and \
                ev["onset"] - (rows[-1]["onset"] + rows[-1]["duration"]) <= gap:
            rows[-1]["duration"] = ev["onset"] + ev["duration"] - rows[-1]["onset"]
        else:
            rows.append({"onset": ev["onset"], "duration": ev["duration"], "trial_type": ev["trial_type"]})
    return pd.DataFrame(rows)


def fit_first_level(boldfile, events: pd.DataFrame, tr: float, mask_img, nuisance=None, glm: dict = None):
    """nilearn FirstLevelModel on one run, with the run's confounds and censored volumes."""
    model = FirstLevelModel(t_r=tr, mask_img=mask_img, **first_level_kwargs(glm))
    confounds = None if nuisance is None else nuisance.confounds
    sample_masks = None if nuisance is None or nuisance.sample_mask is None else [nuisance.sample_mask]
    model.fit(boldfile, events=events[["onset", "duration", "trial_type"]], confounds=confounds,
              sample_masks=sample_masks)
    return model


def contrast_map(model, regressor: str, output_type: str = "t"):
    """Unit contrast on one design column of a fitted FirstLevelModel -> image."""
    columns = list(model.design_matrices_[0].columns)
    vec = np.zeros(len(columns))
    vec[columns.index(regressor)] = 1.0
    return model.compute_contrast(vec, output_type=OUTPUT_TYPES[output_type])


def contrast_values(contrast, output_type: str = "t") -> np.ndarray:
    """A nilearn Contrast -> its t / z / beta values as a flat array."""
    getter = {"t": contrast.stat, "z": contrast.z_score, "beta": contrast.effect_size}[output_type]
    return np.ravel(getter())


def lss(data: np.ndarray, keep: np.ndarray, times: np.ndarray, events: pd.DataFrame, targets,
        confounds: pd.DataFrame = None, glm: dict = None, lss_other: str = "condition"):
    """Least-squares-separate (Mumford et al., 2012): one GLM per target event, with that event
    as the only regressor of interest. The other events are modeled by their trial_type
    ("condition", Mumford's LSS-N) or as one regressor ("single", Mumford's original LSS).

    data    kept volumes x voxels, already masked / smoothed / scaled (see timeseries.glm_ready)
    keep    indices of the kept volumes; designs are built on the full timeline (`times`) and
            censored rows are then dropped, as FirstLevelModel does
    events  onset / duration / trial_type; `targets` are row positions in it
    Yields (row position, nilearn Contrast of the target regressor)."""
    events = events.reset_index(drop=True)
    base = ["lss_other"] * len(events) if lss_other == "single" else list(events["trial_type"])
    kwargs = design_kwargs(glm)
    noise_model = (glm or {}).get("noise_model", "ar1")
    for k in targets:
        names = list(base)
        names[k] = "lss_target"
        ev = pd.DataFrame({"onset": events["onset"].astype(float), "duration": events["duration"].astype(float),
                           "trial_type": names})
        design = make_first_level_design_matrix(times, ev, add_regs=confounds, **kwargs)
        labels, results = run_glm(data, design.values[keep], noise_model=noise_model)
        yield k, compute_contrast(labels, results, (design.columns == "lss_target").astype(float), stat_type="t")


def fit_concatenated(runs: list, shared_prefix: str, glm: dict = None):
    """One GLM over several runs, as SPM fits a multi-session model: regressors whose name starts
    with `shared_prefix` are shared across runs; every other column (nuisance events, drift,
    intercept, confounds) is made run-specific. Handles conditions that occur in only some runs.

    runs  [{"data": kept volumes x voxels (timeseries.glm_ready), "keep": kept indices,
            "n_frames": int, "tr": float, "events": onset/duration/trial_type, "confounds": df|None}]
    -> (labels, results, design columns) for contrast()."""
    glm = glm or {}
    kwargs = design_kwargs(glm)
    designs, data = [], []
    for k, run in enumerate(runs):
        times = frame_times(run["n_frames"], run["tr"], glm.get("slice_time_ref", 0.0))
        design = make_first_level_design_matrix(times, run["events"], add_regs=run.get("confounds"), **kwargs)
        design = design.rename(columns={c: f"run{k}_{c}" for c in design.columns if not str(c).startswith(shared_prefix)})
        designs.append(design.iloc[run["keep"]].reset_index(drop=True))
        data.append(run["data"])
    design = pd.concat(designs, ignore_index=True).fillna(0.0)
    labels, results = run_glm(np.vstack(data), design.values, noise_model=glm.get("noise_model", "ar1"))
    return labels, results, list(design.columns)


def contrast(labels, results, columns: list, weights: dict, stat: str = "t") -> np.ndarray:
    """{column: weight} contrast of a run_glm / fit_concatenated fit -> t or z values."""
    vec = np.zeros(len(columns))
    for col, w in weights.items():
        vec[columns.index(col)] = w
    con = compute_contrast(labels, results, vec, stat_type="t")
    return np.ravel(con.stat() if stat == "t" else con.z_score())


def versus_others(target: str, others) -> dict:
    """Contrast weights: +1 target, -1/len(others) each other (target vs. the mean of the others)."""
    others = [o for o in others if o != target]
    return {target: 1.0, **{o: -1.0 / len(others) for o in others}}
