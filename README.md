# Script Overview

| Script Name        | Input                                                                                     | Output                                    | Function                                                                                      |
|--------------------|-------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------------|
| `compute_mean.py`  | 3 main directories each containing a "volumes" subfolder with 10 brain volumes (`gehirn1.npy` to `gehirn10.npy`) | 1 target directory with 10 mean volumes (`gehirn1.npy` to `gehirn10.npy`) | Computes the voxel-wise mean of volumes from sagittal, coronal, and axial views for each brain |
| `fuseuncertainty.py` | Directory containing sets of 3 super-resolved volumes (`*_sr.npy`) and their uncertainty maps (`*_uncertainty.npy`) per brain | Fused volume with lowest uncertainty per voxel saved in a `fusion_output` subfolder | Performs voxel-wise fusion selecting the voxel value with lowest uncertainty among three views |

