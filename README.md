# betaestimation

Shared fMRI beta and pattern estimation for the lab's analysis workflows, so that decoding and RSA fit identical models from the same preprocessed data:

- [mvpa-decoding](https://github.com/intermountainneuroimaging/mvpa-decoding): classification and timecourse decoding
- rsaworkflow: RDM-based and template RSA

The package covers what happens between fMRIPrep and a pattern analysis:

| Module | What it does |
|---|---|
| `betaestimation.nuisance` | Builds the per-run nuisance model. Confounds come from fMRIPrep, via nilearn `load_confounds` with any strategy (motion 6/12/24, WM/CSF, CompCor, cosines, ICA-AROMA, global signal), or from explicit TSV columns. Censoring covers scrubbing (FD/DVARS), non-steady-state volumes and dropped leading volumes, as a sample mask or as spike regressors. It also writes per-run QC (volumes flagged, number of confounds, FD) and checks the config. |
| `betaestimation.timeseries` | Masks a run, then cleans it: confound regression, detrend, band-pass and z-scoring. Censored volumes are kept in place as NaN rows, so volume indices keep their timing. Also: GLM mean-scaling, and the onset/duration → volume math and event-window syntax from mvpa-decoding. |
| `betaestimation.glm` | nilearn first-level fits with confounds and censoring; least-squares-separate (LSS) per-item fits; one concatenated GLM across runs with shared regressors (SPM-style multi-session); contrasts; mini-block merging. |
| `betaestimation.estimate` | `estimate_run_patterns`: one pattern per item (one event of interest) or per label (its items pooled) for a run. Methods: a pooled GLM, LSA, LSS, or averaging lag-shifted volumes. |
| `betaestimation.masks` | A brain mask shared by all of a subject's runs (one file, an intersection, or an EPI mask); label-image ROIs such as fMRIPrep's aparcaseg, resampled to the BOLD grid; cluster-extent filtering. |

The package doesn't depend on either workflow's config format. Functions take a BOLD file, an events table and plain parameter dicts, with the same keys rsaworkflow's config uses.

## Install

```bash
pip install -e /path/to/betaestimation
```

```bash
pip install -e "/path/to/betaestimation[dev]"
```

The second command adds pytest, to run the tests.

## Use

```python
from betaestimation import estimate_run_patterns, run_nuisance

nuisance = run_nuisance(
    bold, n_frames,
    confounds={"source": "fmriprep", "strategy": ["motion", "high_pass", "wm_csf", "scrub"],
               "motion": "full", "min_segment_volumes": 0, "fd_threshold": 0.5, "std_dvars_threshold": 1.5},
    censoring={"mode": "censor"})

# events: onset, duration, trial_type, label (NaN = not of interest -> nuisance regressor), event_id
for est in estimate_run_patterns(bold, events, tr, n_frames, mask_img, nuisance,
                                 method="glm", unit="item",
                                 glm={"item_model": "lss", "lss_other": "by_label", "output_type": "beta",
                                      "hrf_model": "glover", "drift_model": None, "noise_model": "ar1"}):
    est.img.to_filename(f"beta_{est.label}_{est.event_id}.nii.gz")
```

Vocabulary: an *event* is one row of the events table; an *item* is one event of interest (a row with a `label`); a *label* (an exemplar, or a condition such as a category) groups items for pooled estimates.

The `glm` dict takes any nilearn `FirstLevelModel` keyword, plus `output_type` (`t`, `z` or `beta`), `item_model` (`lsa` or `lss`) and `lss_other` (`by_label` or `single`). For `method="volume_average"`, the `volume_average` dict takes `hemodynamic_lag`, `detrend`, `standardize`, `high_pass`, `low_pass` and `smoothing_fwhm`.

Lower-level pieces:

- **A cleaned time series for TR-wise decoding:**

  ```python
  masker, raw = load_masked(bold, mask, tr)
  clean = clean_run(raw, tr, nuisance, high_pass=1/128)
  ```

  Censored volumes come back as NaN rows.
- **A multi-run GLM with exemplar regressors:**

  ```python
  fit_concatenated(runs, "exm_", glm)
  ```

  Then `contrast(..., versus_others("exm_a", exemplars))` gives an exemplar-vs-others t-map.

## Used by

- **rsaworkflow:** `workflows/estimate_patterns.py` (RDM pattern estimation), `workflows/template_rsa.py` (feature-selection GLMs and time series), and `utils/preprocessing.py`, which adapts the config.
- **mvpa-decoding:** not wired in yet. Natural entry points:
  - `run_nuisance` + `clean_run` for its TR-wise inputs, to replace FSL-only cleaning
  - `estimate_run_patterns(unit="item", glm={"item_model": "lss"})` for beta-series classification
  - `masks.roi_mask` for atlas-label masks

## Tests

```bash
python -m pytest
```

The synthetic runs have known label patterns, written with fMRIPrep naming. The tests cover:

- volume and window math
- cleaning with censored volumes
- fMRIPrep and file confounds, censoring and spike regressors
- masks, label ROIs and cluster filtering
- pattern recovery by pooled GLM, LSA, LSS and volume averaging
- LSS ≡ pooled GLM when each label occurs once per run
- concatenated GLMs with labels present in only some runs
