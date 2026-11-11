"""
One activity pattern per item, or per label (items pooled), for one run.

Vocabulary: an event is one row of the events table; an item is one event of interest (a row
with a label); a label groups items -- an exemplar's repeats, or a condition -- for pooling.

events: a DataFrame with
  onset, duration   seconds
  trial_type        the event's own name (used for nuisance regressors)
  label             what the item is (exemplar or condition name), or NaN for events that are
                    not items (they are modeled as nuisance when model_unlabeled_events)
  event_id          unique per event (e.g. event_index); needed for unit="item"

method "glm":
  unit "pooled"     one regressor per label (all its items), unit contrast -> t / z / beta map
  unit "item"       glm.item_model "lsa": every item its own regressor in one GLM;
                    "lss": one GLM per item (glm.lss_other: other events by label / trial_type,
                    or all in one regressor). The run is masked / smoothed / scaled once.
method "volume_average":
  the run is cleaned (confounds, detrend, band-pass, z-score), then each item's volumes
  (onset + hemodynamic_lag over its duration; censored volumes skipped) are averaged; unit
  "pooled" then averages the items of each label.

Yields PatternEstimate(label, event_id or None, n_events, image).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from betaestimation.glm import check_glm_options, contrast_map, contrast_values, fit_first_level, frame_times, lss, safe_name
from betaestimation.timeseries import clean_run, event_volumes, glm_ready, load_masked

METHODS = {"glm", "volume_average"}
UNITS = {"pooled", "item"}
RENAMED_UNITS = {"condition": "pooled", "trial": "item"}
VOLUME_AVERAGE_DEFAULTS = {"hemodynamic_lag": 4.0, "detrend": True, "standardize": "zscore_sample",
                           "high_pass": None, "low_pass": None, "smoothing_fwhm": None}


@dataclass
class PatternEstimate:
    label: object
    event_id: object
    n_events: int
    img: object


def _regressor(row, unit: str) -> str:
    if pd.isna(row["label"]):
        return f"nuisance_{safe_name(row['trial_type'])}"
    if unit == "item":
        return f"lbl_{safe_name(row['label'])}_ev{safe_name(row['event_id'])}"
    return f"lbl_{safe_name(row['label'])}"


def check_label_names(labels):
    """Regressors are named lbl_<safe_name(label)>, which keeps only letters and digits, so labels
    that differ only in other characters (dog_1 / dog1) would be fit as ONE regressor."""
    seen = {}
    for label in labels:
        seen.setdefault(safe_name(label), []).append(label)
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    if clashes:
        detail = "; ".join(f"{v} -> {k!r}" for k, v in clashes.items())
        raise ValueError(f"labels collide after removing non-alphanumeric characters: {detail}")


def estimate_run_patterns(boldfile, events: pd.DataFrame, tr: float, n_frames: int, mask_img, nuisance=None,
                          method: str = "glm", unit: str = "pooled", glm: dict = None,
                          volume_average: dict = None, model_unlabeled_events: bool = True):
    if unit in RENAMED_UNITS:
        raise ValueError(f"unit {unit!r} has been renamed {RENAMED_UNITS[unit]!r}")
    if method not in METHODS or unit not in UNITS:
        raise ValueError(f"method must be one of {sorted(METHODS)} and unit one of {sorted(UNITS)}")
    events = events.reset_index(drop=True)
    check_label_names(events["label"].dropna().unique())
    if method == "volume_average":
        yield from _volume_average(boldfile, events, tr, n_frames, mask_img, nuisance, unit,
                                   {**VOLUME_AVERAGE_DEFAULTS, **(volume_average or {})})
        return
    glm = dict(glm or {})
    problems = check_glm_options(glm)
    if problems:
        raise ValueError("; ".join(problems))
    output_type = glm.get("output_type", "t")
    model_df = events if model_unlabeled_events else events.dropna(subset=["label"]).reset_index(drop=True)
    if unit == "item" and glm.get("item_model", "lss") == "lss":
        yield from _lss(boldfile, model_df, tr, n_frames, mask_img, nuisance, glm, output_type)
        return

    design_events = model_df.assign(trial_type=[_regressor(r, unit) for _, r in model_df.iterrows()])
    model = fit_first_level(boldfile, design_events, tr, mask_img, nuisance, glm)
    labeled = model_df.dropna(subset=["label"])
    groups = labeled.groupby("event_id", sort=False) if unit == "item" else labeled.groupby("label", sort=False)
    for _, grp in groups:
        first = grp.iloc[0]
        img = contrast_map(model, _regressor(first, unit), output_type)
        yield PatternEstimate(first["label"], first["event_id"] if unit == "item" else None, len(grp), img)


def _lss(boldfile, model_df, tr, n_frames, mask_img, nuisance, glm, output_type):
    masker, raw = load_masked(boldfile, mask_img, tr, glm.get("smoothing_fwhm"))
    keep = np.arange(n_frames) if nuisance is None else nuisance.kept(n_frames)
    data = glm_ready(raw, nuisance, glm.get("signal_scaling", 0))
    times = frame_times(raw.shape[0], tr, glm.get("slice_time_ref", 0.0))
    events = model_df.assign(trial_type=[_regressor(r, "pooled") for _, r in model_df.iterrows()])
    targets = list(np.flatnonzero(model_df["label"].notna().values))
    confounds = None if nuisance is None else nuisance.confounds
    for k, con in lss(data, keep, times, events, targets, confounds, glm, glm.get("lss_other", "by_label")):
        row = model_df.iloc[k]
        yield PatternEstimate(row["label"], row["event_id"], 1, masker.inverse_transform(contrast_values(con, output_type)))


def _volume_average(boldfile, events, tr, n_frames, mask_img, nuisance, unit, va):
    masker, raw = load_masked(boldfile, mask_img, tr, va.get("smoothing_fwhm"))
    data = clean_run(raw, tr, nuisance, detrend=va["detrend"], standardize=va["standardize"],
                     high_pass=va["high_pass"], low_pass=va.get("low_pass"))
    per_event = []
    for _, ev in events.dropna(subset=["label"]).iterrows():
        vols = event_volumes(float(ev["onset"]), float(ev["duration"]), tr, n_frames, va["hemodynamic_lag"])
        if len(vols) == 0:
            print(f"  (!) event {ev['trial_type']} at {ev['onset']}s falls past the end of the run; skipped")
            continue
        window = data[vols]
        if np.isnan(window[:, 0]).all():
            print(f"  (!) every volume of event {ev['trial_type']} at {ev['onset']}s is censored; skipped")
            continue
        per_event.append((ev["label"], ev["event_id"], np.nanmean(window, axis=0)))

    if unit == "item":
        for label, event_id, vec in per_event:
            yield PatternEstimate(label, event_id, 1, masker.inverse_transform(vec))
        return
    for label in dict.fromkeys(lbl for lbl, _, _ in per_event):
        vecs = [v for lbl, _, v in per_event if lbl == label]
        yield PatternEstimate(label, None, len(vecs), masker.inverse_transform(np.mean(vecs, axis=0)))
