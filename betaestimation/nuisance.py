"""
Per-run nuisance model: confound regressors, volume censoring (scrubbing / dummy scans), and a
per-run QC summary.

confounds (a dict, or None for no confound regressors):
  {"source": "fmriprep", ...}            nilearn.interfaces.fmriprep.load_confounds on the run's
                                         fMRIPrep BOLD file; every other key is passed through
                                         (strategy, motion, wm_csf, global_signal, compcor,
                                         n_compcor, fd_threshold, std_dvars_threshold,
                                         ica_aroma, tedana, demean). With "scrub" /
                                         "non_steady_state" in the strategy it also returns which
                                         volumes to keep.
     min_segment_volumes                 (only with "scrub" in the strategy) after the FD / DVARS
                                         outliers are flagged, runs of good volumes shorter than
                                         this, stranded between outliers or at a run edge, are
                                         flagged too. 0 = flag only the outliers. Omitted = nilearn's
                                         default of 5 (Power et al. 2014). Passed to nilearn as its
                                         `scrub` argument, renamed here because "scrub" is also the
                                         strategy entry that switches outlier detection on.
  {"source": "file", "file_pattern",     explicit columns (and/or regex-matched columns) from any
   "columns": [...], "regex": [...]}     TSV; {subject} / {session} / {task} / {run} are filled from
                                         `entities`.

censoring (a dict, or None):
  drop_initial_volumes  N leading volumes to discard (dummy scans not already flagged by fMRIPrep)
  mode                  "censor": flagged volumes are removed from fits (sample mask)
                        "regress": one spike regressor per flagged volume; the time series stays
                                   continuous (some prefer this with AR prewhitening)
"""

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONFOUND_SOURCES = {"fmriprep", "file"}
CENSORING_MODES = {"censor", "regress"}


@dataclass
class RunNuisance:
    confounds: pd.DataFrame = None   # n_frames rows, or None
    sample_mask: np.ndarray = None   # indices of volumes kept, or None for "all"
    qc: dict = field(default_factory=dict)

    def kept(self, n_frames: int) -> np.ndarray:
        return np.arange(n_frames) if self.sample_mask is None else self.sample_mask


class _KeepMissing(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def format_entities(template: str, entities: dict) -> str:
    """Fill {subject} / {session} / {task} / {run} placeholders; unknown ones are left intact."""
    return template.format_map(_KeepMissing({k: v for k, v in (entities or {}).items() if v not in (None, "")}))


def fmriprep_confounds_path(boldfile: str) -> str:
    """fMRIPrep's confounds TSV for a preprocessed BOLD file: same entities up to space/res/den/desc."""
    name = os.path.basename(boldfile)
    stem = re.split(r"_(?:space|res|den|desc)-", name)[0].replace("_bold.nii.gz", "").replace("_bold.nii", "")
    return os.path.join(os.path.dirname(boldfile), f"{stem}_desc-confounds_timeseries.tsv")


def load_file_confounds(spec: dict, entities: dict) -> pd.DataFrame:
    path = format_entities(spec["file_pattern"], entities)
    matches = glob.glob(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"confounds: expected 1 file for {path!r}, found {len(matches)}")
    conf = pd.read_csv(matches[0], sep="\t")
    cols = list(spec.get("columns") or [])
    for rx in spec.get("regex") or []:
        cols += [c for c in conf.columns if re.fullmatch(rx, c) and c not in cols]
    missing = [c for c in cols if c not in conf.columns]
    if missing:
        raise KeyError(f"confounds: column(s) {missing} not in {matches[0]}")
    return conf[cols].fillna(0.0)


MIN_SEGMENT_KEY = "min_segment_volumes"


def fmriprep_kwargs(confounds: dict) -> dict:
    """The confounds spec as nilearn load_confounds arguments: min_segment_volumes -> scrub."""
    if "scrub" in confounds:
        raise ValueError(f"confounds: the 'scrub' key is now {MIN_SEGMENT_KEY!r} (keep 'scrub' in the strategy list)")
    kwargs = {k: (tuple(v) if k == "strategy" else v) for k, v in confounds.items()
              if k not in ("source", MIN_SEGMENT_KEY)}
    if MIN_SEGMENT_KEY in confounds:
        kwargs["scrub"] = int(confounds[MIN_SEGMENT_KEY])
    return kwargs


def run_nuisance(boldfile: str, n_frames: int, confounds: dict = None, censoring: dict = None,
                 entities: dict = None) -> RunNuisance:
    """Confound regressors, kept volumes and QC for one run.

    `entities` ({subject, session, task, run}) fill a file-confounds pattern; they are not needed
    for fMRIPrep confounds, which are found next to the BOLD file."""
    censoring = censoring or {}
    conf, keep = None, None
    if confounds:
        if confounds.get("source", "fmriprep") == "fmriprep":
            from nilearn.interfaces.fmriprep import load_confounds
            conf, keep = load_confounds(boldfile, **fmriprep_kwargs(confounds))
        else:
            conf = load_file_confounds(confounds, entities)

    drop = int(censoring.get("drop_initial_volumes") or 0)
    if drop:
        keep = np.arange(drop, n_frames) if keep is None else keep[keep >= drop]
    if keep is not None:
        keep = np.asarray(keep, dtype=int)
        if len(keep) == n_frames:
            keep = None

    flagged = np.setdiff1d(np.arange(n_frames), keep) if keep is not None else np.array([], dtype=int)
    if keep is not None and censoring.get("mode", "censor") == "regress":
        spikes = pd.DataFrame(np.eye(n_frames)[:, flagged], columns=[f"spike_{v:04d}" for v in flagged])
        conf = spikes if conf is None else pd.concat([conf.reset_index(drop=True), spikes], axis=1)
        keep = None
    if conf is not None:
        conf = conf.reset_index(drop=True).fillna(0.0)
        if len(conf) != n_frames:
            raise ValueError(f"{boldfile}: {len(conf)} confound rows but {n_frames} volumes")

    qc = {"n_volumes": n_frames, "n_flagged": len(flagged), "pct_flagged": round(100 * len(flagged) / n_frames, 1),
          "censoring": censoring.get("mode", "censor") if len(flagged) else "",
          "n_confounds": 0 if conf is None else conf.shape[1]}
    tsv = fmriprep_confounds_path(boldfile)
    if os.path.exists(tsv):
        fd = pd.read_csv(tsv, sep="\t", usecols=lambda c: c == "framewise_displacement")
        if "framewise_displacement" in fd:
            qc["mean_fd"] = round(float(fd["framewise_displacement"].mean()), 3)
            qc["max_fd"] = round(float(fd["framewise_displacement"].max()), 3)
    return RunNuisance(conf, keep, qc)


def validate_nuisance(confounds: dict = None, censoring: dict = None, drift_model=None) -> tuple:
    """Static checks -> (errors, warnings). `drift_model` is the GLM's drift model, if a GLM is
    fit, to catch a second high-pass set on top of fMRIPrep's cosine regressors."""
    errors, warnings = [], []
    if confounds:
        source = confounds.get("source", "fmriprep")
        if source not in CONFOUND_SOURCES:
            errors.append(f"confounds.source must be one of {sorted(CONFOUND_SOURCES)}")
        elif source == "file" and not confounds.get("file_pattern"):
            errors.append("confounds: source 'file' needs 'file_pattern'")
        elif source == "fmriprep":
            strategy = list(confounds.get("strategy", ("motion", "high_pass", "wm_csf")))
            if "high_pass" in strategy and drift_model:
                warnings.append("confounds.strategy includes fMRIPrep's 'high_pass' cosine regressors AND the GLM's "
                                f"drift_model={drift_model!r}: the two drift sets are near-collinear; set drift_model "
                                "to null or drop 'high_pass' from the strategy")
            if "compcor" in strategy and "high_pass" not in strategy:
                errors.append("confounds.strategy: fMRIPrep's CompCor components assume the 'high_pass' "
                              "cosine regressors are also included (nilearn enforces this)")
            if "scrub" in confounds:
                errors.append(f"confounds.scrub has been renamed confounds.{MIN_SEGMENT_KEY} (the minimum run of "
                              "good volumes kept between outliers); 'scrub' stays in the strategy list")
            seg = confounds.get(MIN_SEGMENT_KEY)
            if seg is not None:
                if isinstance(seg, bool) or not isinstance(seg, int) or seg < 0:
                    errors.append(f"confounds.{MIN_SEGMENT_KEY} must be an integer >= 0")
                elif "scrub" not in strategy:
                    warnings.append(f"confounds.{MIN_SEGMENT_KEY} has no effect without 'scrub' in confounds.strategy")
            if "global_signal" in strategy:
                warnings.append("confounds.strategy includes 'global_signal'; GSR can induce negative correlations "
                                "and shift pattern similarity -- make sure that is intended")
    if (censoring or {}).get("mode", "censor") not in CENSORING_MODES:
        errors.append(f"censoring.mode must be one of {sorted(CENSORING_MODES)}")
    return errors, warnings


def missing_confound_files(confounds: dict, boldfiles) -> list:
    """With source 'fmriprep', the BOLD files whose desc-confounds_timeseries.tsv is missing."""
    if not confounds or confounds.get("source", "fmriprep") != "fmriprep":
        return []
    return [b for b in sorted(set(boldfiles)) if not os.path.exists(fmriprep_confounds_path(b))]
