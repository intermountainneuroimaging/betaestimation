"""
Masks: a subject brain mask shared by every run, label-image ROIs (e.g. fMRIPrep aparcaseg), and
cluster-extent filtering of thresholded maps.
"""

import nibabel as nib
import numpy as np
from nilearn.image import mean_img, resample_to_img
from nilearn.masking import compute_epi_mask, intersect_masks
from scipy import ndimage


def brain_mask(mask_files=None, reference_bold=None):
    """One brain mask for all of a subject's runs, so every pattern has the same voxels:
    the single file given, the intersection of several (e.g. fMRIPrep's per-run desc-brain_mask),
    or -- with no files -- an EPI mask computed from the mean of reference_bold.
    -> (mask image, how it was made: "file" / "intersection" / "epi")."""
    files = list(mask_files or [])
    if len(files) == 1:
        return nib.load(files[0]), "file"
    if files:
        return intersect_masks(files, threshold=1, connected=False), "intersection"
    if reference_bold is None:
        raise ValueError("brain_mask: give mask files or a reference BOLD file")
    return compute_epi_mask(mean_img(reference_bold, copy_header=True)), "epi"


def roi_mask(path: str, labels=None, reference_img=None):
    """Binary ROI from a mask image (non-zero voxels) or a label image (the union of `labels`,
    e.g. Desikan-Killiany ids from fMRIPrep's aparcaseg). With reference_img, resampled
    (nearest neighbour) onto its grid -- e.g. a 1 mm atlas onto the 2.4 mm BOLD grid."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    data = np.isin(np.round(data).astype(int), list(labels)) if labels else data > 0
    mask = nib.Nifti1Image(data.astype(np.uint8), img.affine)
    if reference_img is not None:
        ref = nib.load(reference_img) if isinstance(reference_img, str) else reference_img
        if ref.shape[:3] != mask.shape or not np.allclose(ref.affine, mask.affine):
            mask = resample_to_img(mask, ref, interpolation="nearest", force_resample=True, copy_header=True)
    return mask


def cluster_filter(significant: np.ndarray, masker, min_size: int) -> np.ndarray:
    """Keep only voxels of `significant` (a boolean vector in masker space) that belong to
    26-connected clusters of at least min_size voxels."""
    if min_size <= 1 or not significant.any():
        return significant
    vol = np.asarray(masker.inverse_transform(significant.astype(np.float32)).dataobj) > 0
    lab, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(vol, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, np.flatnonzero(sizes >= min_size) + 1)
    return masker.transform(nib.Nifti1Image(keep.astype(np.float32), masker.mask_img_.affine)).ravel() > 0.5
