"""
betaestimation -- shared fMRI beta / pattern estimation for the lab's MVPA (mvpa-decoding) and RSA
(rsaworkflow) workflows.

  nuisance     fMRIPrep / TSV confounds, censoring (scrub, dummy scans, spike regressors), run QC
  timeseries   masking, cleaning with censored volumes kept as NaN rows, GLM scaling, event windows
  glm          first-level fits, LSS, concatenated multi-run GLMs, contrasts, mini-block merging
  estimate     one pattern per condition or trial per run (GLM: condition / LSA / LSS; volume averaging)
  masks        subject brain mask, label-image ROIs, cluster-extent filtering
"""

from betaestimation.estimate import PatternEstimate, estimate_run_patterns
from betaestimation.nuisance import RunNuisance, run_nuisance, validate_nuisance

__version__ = "0.1.0"
__all__ = ["PatternEstimate", "RunNuisance", "estimate_run_patterns", "run_nuisance", "validate_nuisance"]
