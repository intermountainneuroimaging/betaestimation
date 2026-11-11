import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.maskers import NiftiMasker

from betaestimation.estimate import estimate_run_patterns
from betaestimation.glm import contrast, fit_concatenated, merge_blocks, versus_others
from betaestimation.masks import brain_mask, cluster_filter, roi_mask
from betaestimation.nuisance import fmriprep_confounds_path, missing_confound_files, run_nuisance, validate_nuisance
from betaestimation.timeseries import (
    clean_run, compute_volume_range, event_volumes, glm_ready, load_masked, window_volumes)
from tests.conftest import AFFINE, SHAPE, TR, events_table, make_run, patterns

FMRIPREP = {"source": "fmriprep", "strategy": ["motion", "scrub"], "motion": "basic", "min_segment_volumes": 0,
            "fd_threshold": 0.5, "std_dvars_threshold": 3.0}


# ---------------------------------------------------------------- timeseries

def test_volume_math_matches_mvpa_decoding():
    assert compute_volume_range(4.0, 28.0, 2.5, 121) == (1, 11)
    assert compute_volume_range(300.0, 330.0, 2.5, 121) == (120, 121)
    assert compute_volume_range(0.2, 0.4, 2.0, 10) == (0, 1)
    assert list(event_volumes(10.0, 4.0, 2.0, 100, lag=4.0)) == [7, 8]
    win = {"start": {"reference": "onset", "offset_seconds": 4.0}, "end": {"reference": "offset_end", "offset_seconds": 4.0}}
    assert list(window_volumes(win, 10.0, 4.0, 2.0, 100)) == [7, 8]


def test_clean_run_keeps_censored_volumes_as_nan(run_files):
    nuis = run_nuisance(run_files["bold"], 90, FMRIPREP)
    _, raw = load_masked(run_files["bold"], run_files["mask"], TR)
    clean = clean_run(raw, TR, nuis, high_pass=0.01)
    assert clean.shape == raw.shape
    censored = np.setdiff1d(np.arange(90), nuis.sample_mask)
    assert {30, 31} <= set(censored) and np.isnan(clean[censored]).all()
    kept = clean[nuis.sample_mask]
    assert np.allclose(kept.mean(axis=0), 0, atol=1e-6) and np.allclose(kept.std(axis=0, ddof=1), 1, atol=1e-6)
    scaled = glm_ready(raw, nuis, 0)
    assert scaled.shape[0] == len(nuis.sample_mask) and abs(scaled.mean()) < 1e-6


# ---------------------------------------------------------------- nuisance

def test_nuisance_censor_regress_and_drop(run_files):
    assert fmriprep_confounds_path(run_files["bold"]).endswith("sub-01_task-x_run-01_desc-confounds_timeseries.tsv")
    censor = run_nuisance(run_files["bold"], 90, FMRIPREP)
    assert censor.confounds.shape == (90, 6) and censor.qc["n_flagged"] >= 2 and censor.qc["mean_fd"] > 0
    regress = run_nuisance(run_files["bold"], 90, FMRIPREP, {"mode": "regress", "drop_initial_volumes": 3})
    assert regress.sample_mask is None
    assert len([c for c in regress.confounds if c.startswith("spike_")]) == regress.qc["n_flagged"] >= 5
    drop_only = run_nuisance(run_files["bold"], 90, None, {"drop_initial_volumes": 4})
    assert drop_only.confounds is None and drop_only.sample_mask[0] == 4


def test_min_segment_volumes_also_flags_short_good_segments(run_files):
    exact = run_nuisance(run_files["bold"], 90, FMRIPREP)
    padded = run_nuisance(run_files["bold"], 90, {**FMRIPREP, "min_segment_volumes": 50})
    assert padded.qc["n_flagged"] > exact.qc["n_flagged"]   # outliers plus the short segments around them
    assert set(padded.sample_mask) < set(exact.sample_mask)
    with pytest.raises(ValueError, match="min_segment_volumes"):
        run_nuisance(run_files["bold"], 90, {"source": "fmriprep", "strategy": ["motion", "scrub"], "scrub": 0})
    assert any("renamed" in e for e in validate_nuisance({"strategy": ["scrub"], "scrub": 0})[0])
    assert any("integer" in e for e in validate_nuisance({"strategy": ["scrub"], "min_segment_volumes": -1})[0])
    assert any("no effect" in w for w in validate_nuisance({"strategy": ["motion"], "min_segment_volumes": 0})[1])


def test_file_confounds_use_entities(run_files):
    spec = {"source": "file", "file_pattern": str(run_files["dir"] / "sub-{subject}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"),
            "columns": ["csf"], "regex": ["trans_."]}
    nuis = run_nuisance(run_files["bold"], 90, spec, entities={"subject": "01", "task": "x", "run": "01"})
    assert list(nuis.confounds.columns) == ["csf", "trans_x", "trans_y", "trans_z"]


def test_validate_nuisance_and_missing_files(run_files, tmp_path):
    errors, warnings = validate_nuisance({"source": "fmriprep", "strategy": ["motion", "high_pass", "compcor"]},
                                         drift_model="cosine")
    assert errors == [] and any("near-collinear" in w for w in warnings)
    assert any("CompCor" in e for e in validate_nuisance({"strategy": ["compcor"]})[0])
    assert any("censoring.mode" in e for e in validate_nuisance(None, {"mode": "drop"})[0])
    assert missing_confound_files(FMRIPREP, [run_files["bold"], str(tmp_path / "other_bold.nii.gz")]) == \
        [str(tmp_path / "other_bold.nii.gz")]


# ---------------------------------------------------------------- masks

def test_brain_mask_and_roi_mask(tmp_path, run_files):
    a, b = tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"
    da = np.ones(SHAPE, np.uint8)
    db = np.ones(SHAPE, np.uint8)
    db[0] = 0
    nib.Nifti1Image(da, AFFINE).to_filename(a)
    nib.Nifti1Image(db, AFFINE).to_filename(b)
    mask, how = brain_mask([a, b])
    assert how == "intersection" and np.asarray(mask.dataobj).sum() == db.sum()
    assert brain_mask(None, run_files["bold"])[1] == "epi"

    labels = np.zeros((12, 12, 12), np.int16)
    labels[:6] = 1007
    labels[6:] = 17
    atlas = tmp_path / "dseg.nii.gz"
    nib.Nifti1Image(labels, np.diag([1.5, 1.5, 1.5, 1.0])).to_filename(atlas)  # finer grid than the BOLD
    roi = roi_mask(str(atlas), [1007], run_files["bold"])
    assert roi.shape == SHAPE
    assert 0 < np.asarray(roi.dataobj).sum() < np.prod(SHAPE)


def test_cluster_filter_drops_small_clusters(run_files):
    masker = NiftiMasker(mask_img=run_files["mask"]).fit()
    vol = np.zeros(SHAPE, bool)
    vol[:3, :3, :3] = True     # 27 voxels
    vol[5, 5, 5] = True        # isolated voxel
    sig = masker.transform(nib.Nifti1Image(vol.astype(np.float32), AFFINE)).ravel() > 0.5
    kept = cluster_filter(sig, masker, 10)
    assert kept.sum() == 27


# ---------------------------------------------------------------- estimation

def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def recovered(estimates, pats, masker):
    return {e.label: corr(masker.transform(e.img).ravel(), pats[e.label]) for e in estimates}


@pytest.mark.parametrize("method,unit,glm", [
    ("glm", "pooled", {"output_type": "beta", "noise_model": "ols"}),
    ("glm", "item", {"output_type": "beta", "item_model": "lsa", "noise_model": "ols"}),
    ("glm", "item", {"output_type": "beta", "item_model": "lss", "noise_model": "ols"}),
    ("volume_average", "pooled", None),
])
def test_estimates_recover_condition_patterns(run_files, method, unit, glm):
    masker = NiftiMasker(mask_img=run_files["mask"]).fit()
    nuis = run_nuisance(run_files["bold"], 90, FMRIPREP)
    est = list(estimate_run_patterns(run_files["bold"], run_files["events"], TR, 90, run_files["mask"], nuis,
                                     method=method, unit=unit, glm=glm))
    n_labeled = run_files["events"]["label"].notna().sum()
    assert len(est) == (3 if unit == "pooled" else n_labeled)
    assert all(e.label in "abc" for e in est)
    if unit == "item":
        assert sorted(e.event_id for e in est) == sorted(run_files["events"].dropna(subset=["label"])["event_id"])
    r = recovered(est, run_files["patterns"], masker)
    assert min(r.values()) > 0.8


def test_lss_equals_condition_glm_when_each_condition_occurs_once(tmp_path):
    """With one event per condition, LSS's design (target + the other conditions) is the
    condition GLM's design, so their maps must match to floating-point precision."""
    pats = patterns(3)
    ev = events_table("abc", [6, 40, 74])
    bold = tmp_path / "sub-01_task-x_run-01_desc-preproc_bold.nii.gz"
    make_run(bold, ev, pats, seed=3)
    mask = tmp_path / "mask.nii.gz"
    nib.Nifti1Image(np.ones(SHAPE, np.uint8), AFFINE).to_filename(mask)
    glm = {"output_type": "t", "noise_model": "ols"}
    cond = {e.label: e.img for e in estimate_run_patterns(str(bold), ev, TR, 90, str(mask), method="glm",
                                                          unit="pooled", glm=glm)}
    lss = {e.label: e.img for e in estimate_run_patterns(str(bold), ev, TR, 90, str(mask), method="glm", unit="item",
                                                         glm={**glm, "item_model": "lss"})}
    for label in "abc":
        assert np.allclose(np.asarray(cond[label].dataobj), np.asarray(lss[label].dataobj), atol=1e-6)


def test_concatenated_glm_with_conditions_in_only_some_runs(tmp_path):
    pats = patterns(5)
    mask = tmp_path / "mask.nii.gz"
    nib.Nifti1Image(np.ones(SHAPE, np.uint8), AFFINE).to_filename(mask)
    runs = []
    for k, labels in enumerate(["ab", "bc"]):   # a only in run 1, c only in run 2
        ev = events_table(labels, [10, 50])
        bold = tmp_path / f"run{k}_bold.nii.gz"
        make_run(bold, ev, pats, seed=10 + k)
        _, raw = load_masked(str(bold), str(mask), TR)
        runs.append({"data": glm_ready(raw), "keep": np.arange(90), "n_frames": 90, "tr": TR,
                     "events": ev.assign(trial_type="item_" + ev["label"])[["onset", "duration", "trial_type"]],
                     "confounds": None})
    labels, results, columns = fit_concatenated(runs, "item_", {"noise_model": "ols"})
    assert {"item_a", "item_b", "item_c"} <= set(columns)
    assert any(c.startswith("run0_") for c in columns) and any(c.startswith("run1_") for c in columns)
    for item in "abc":
        t = contrast(labels, results, columns, versus_others(f"item_{item}", ["item_a", "item_b", "item_c"]))
        others = [pats[o] for o in "abc" if o != item]
        assert corr(t, pats[item] - np.mean(others, axis=0)) > 0.7


def test_merge_blocks():
    ev = pd.DataFrame({"onset": [0, 3, 6, 20], "duration": 1.0, "trial_type": ["x", "x", "y", "x"]})
    merged = merge_blocks(ev, gap=2.5)
    assert merged.to_dict("list") == {"onset": [0, 6, 20], "duration": [4.0, 1.0, 1.0], "trial_type": ["x", "y", "x"]}


def test_labels_that_collide_after_safe_name_are_rejected():
    from betaestimation.estimate import check_label_names
    check_label_names(["dog1", "dog2", "cat-1"])
    with pytest.raises(ValueError, match="dog_1"):
        check_label_names(["dog_1", "dog1", "cat"])


def test_renamed_units_and_glm_keys_are_rejected(run_files):
    from betaestimation.estimate import estimate_run_patterns
    ev = pd.DataFrame({"onset": [10.0], "duration": [2.0], "trial_type": ["a"], "label": ["a"], "event_id": [0]})
    with pytest.raises(ValueError, match="'pooled'"):
        list(estimate_run_patterns(run_files["bold"], ev, 2.0, 90, None, unit="condition"))
    with pytest.raises(ValueError, match="item_model"):
        list(estimate_run_patterns(run_files["bold"], ev, 2.0, 90, None, unit="item", glm={"trial_model": "lss"}))
    with pytest.raises(ValueError, match="by_label"):
        list(estimate_run_patterns(run_files["bold"], ev, 2.0, 90, None, unit="item", glm={"lss_other": "condition"}))


def test_file_confounds_outlier_columns_censor_volumes(run_files, tmp_path):
    src = fmriprep_confounds_path(run_files["bold"])
    conf = pd.read_csv(src, sep="\t")
    for v in (5, 40):
        conf[f"motion_outlier_{v:02d}"] = (conf.index == v).astype(int)
    path = tmp_path / "conf.tsv"
    conf.to_csv(path, sep="\t", index=False)
    spec = {"source": "file", "file_pattern": str(path), "columns": ["trans_x"], "outlier_regex": ["motion_outlier_.*"]}
    censored = run_nuisance(run_files["bold"], 90, spec, {"drop_initial_volumes": 2})
    assert list(censored.confounds.columns) == ["trans_x"]           # outlier columns are not regressors
    assert {0, 1, 5, 40} == set(range(90)) - set(censored.sample_mask) and censored.qc["n_flagged"] == 4
    regressed = run_nuisance(run_files["bold"], 90, spec, {"mode": "regress"})
    assert regressed.sample_mask is None and [c for c in regressed.confounds if c.startswith("spike_")] == ["spike_0005", "spike_0040"]
    assert any("both" in e for e in validate_nuisance({"source": "file", "file_pattern": "x", "regex": ["m.*"],
                                                        "outlier_regex": ["m.*"]})[0])
