"""Synthetic runs with known condition patterns, written as fMRIPrep-style files."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

TR, SHAPE = 2.0, (6, 6, 6)
AFFINE = np.diag([3.0, 3.0, 3.0, 1.0])
N_VOX = int(np.prod(SHAPE))
SPACE = "space-MNI152NLin2009cAsym"


def patterns(seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(N_VOX) for k in "abc"}


def make_run(path, events, pats, n_vols=90, amp=3.0, noise=1.0, seed=0, motion_artifact=None):
    """BOLD = 100 + sum over labeled events of HRF-convolved boxcar x pattern + noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(n_vols) * TR
    data = 100 + noise * rng.standard_normal((n_vols, N_VOX))
    for _, ev in events.iterrows():
        if pd.isna(ev.get("label")):
            continue
        reg = make_first_level_design_matrix(times, pd.DataFrame(
            {"onset": [ev["onset"]], "duration": [ev["duration"]], "trial_type": ["x"]}), drift_model=None)["x"].values
        data += amp * np.outer(reg, pats[ev["label"]])
    if motion_artifact is not None:
        data += motion_artifact
    img = nib.Nifti1Image(data.T.reshape(*SHAPE, n_vols).astype(np.float32), AFFINE)
    img.header.set_zooms((3.0, 3.0, 3.0, TR))
    img.to_filename(path)
    return data


def events_table(labels, onsets, duration=6.0, extra=None):
    ev = pd.DataFrame({"onset": onsets, "duration": duration, "trial_type": list(labels), "label": list(labels)})
    if extra:
        for onset, name in extra:
            ev.loc[len(ev)] = {"onset": onset, "duration": 2.0, "trial_type": name, "label": np.nan}
    ev = ev.sort_values("onset").reset_index(drop=True)
    ev["event_id"] = np.arange(1, len(ev) + 1)
    return ev


@pytest.fixture
def run_files(tmp_path):
    """One 90-volume run: a, b, c twice each (6 s blocks) + a nuisance 'cue' event, a brain mask,
    and fMRIPrep confounds with FD spikes at volumes 30 and 31."""
    pats = patterns()
    ev = events_table("abcabc", [6, 30, 54, 90, 114, 138], extra=[(20, "cue"), (100, "cue")])
    stem = f"sub-01_task-x_run-01"
    bold = tmp_path / f"{stem}_{SPACE}_desc-preproc_bold.nii.gz"
    make_run(bold, ev, pats)
    mask = tmp_path / f"{stem}_{SPACE}_desc-brain_mask.nii.gz"
    nib.Nifti1Image(np.ones(SHAPE, np.uint8), AFFINE).to_filename(mask)
    rng = np.random.default_rng(1)
    conf = pd.DataFrame(rng.standard_normal((90, 6)) * 0.05, columns=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"])
    fd = np.abs(rng.standard_normal(90)) * 0.1
    fd[[30, 31]] = 2.0
    conf["framewise_displacement"] = np.r_[np.nan, fd[1:]]
    conf["std_dvars"] = np.r_[np.nan, np.ones(89)]
    conf["csf"] = rng.standard_normal(90)
    conf.to_csv(tmp_path / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False, na_rep="n/a")
    (tmp_path / f"{stem}_desc-confounds_timeseries.json").write_text(json.dumps({}))
    return {"bold": str(bold), "mask": str(mask), "events": ev, "patterns": pats, "dir": tmp_path}
