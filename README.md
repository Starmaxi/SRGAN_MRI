# Script Overview

| Script Name           | Input                                                                                     | Output                                    | Function                                                                                      |
|-----------------------|-------------------------------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------------|
| `compute_mean.py`     | 3 main directories each containing a "volumes" subfolder with 10 brain volumes (`gehirn1.npy` to `gehirn10.npy`) | 1 target directory with 10 mean volumes (`gehirn1.npy` to `gehirn10.npy`) | Computes the voxel-wise mean of volumes from sagittal, coronal, and axial views for each brain |
| `fuseuncertainty.py`  | Directory containing sets of 3 super-resolved volumes (`*_sr.npy`) and their uncertainty maps (`*_uncertainty.npy`) per brain | Fused volume with lowest uncertainty per voxel saved in a `fusion_output` subfolder | Performs voxel-wise fusion selecting the voxel value with lowest uncertainty among three views |
| `recreate3d2.py`      | Main folder containing multiple subfolders, each with numerically ordered 2D image slices (PNG, JPG, TIFF, etc.) | Numpy volumes saved in a `volumes` subfolder inside main folder, named after each subfolder | Converts ordered 2D image slices into 3D numpy volume files                                  |
| `resizesagittal.ipynb`| Directory with `.npy` volumes in shape (170, 256, 256)                                    | Overwrites input `.npy` files resized to (256, 256, 256) along depth axis       | Resizes sagittal volumes along depth axis from 170 slices to 256 using linear interpolation  |
| `test3drichtig.ipynb` | Multiple `.npy` volume files such as `"gehirn3axial.npy"`, `"gehirn3coronal.npy"`, `"gehirn3sagumsortiert.npy"`       | Visualizations of slices in axial, coronal, sagittal orientations; transposed and saved `.npy` files | Interactive notebook with multiple cells for visualizing 3D MRI volumes, slicing views, volume shape checks, and transposition of volume axes for correct orientation |
| `compute_uncertainty.py` (incomplete) | Folder with 2D grayscale images (PNG) to be super-resolved by a generator model with Monte Carlo dropout  | Super-resolved images and uncertainty maps saved as PNG in output directory   | Calculates super-resolution images with uncertainty maps using Monte Carlo Dropout; **Script incomplete: missing generator model and function call** |
| `test_inferenz.ipynb` | Two grayscale images (`lr_test.png` and `hr_test.png`) resized to 256×256 pixels             | PSNR and SSIM values printed; visual plots of LR, HR, error maps, and histograms | Notebook to compare low-resolution and high-resolution images quantitatively and visually using PSNR, SSIM, error maps, and pixel value histograms |


## Validation script:
## Requirements

```bash
pip install numpy matplotlib scikit-image pillow
```

## Usage

### 1. PSNR and SSIM Calculation

```python
psnr_value = calculate_psnr(original, compared)
ssim_value = calculate_ssim(original_volume, compared_volume, slice_idx=80)
```

### 2. Frequency Domain (FFT) Visualization

```python
fft_volume = get_image_volume_fft2(image_vol)
show_slices_side_by_side(fft_volume, fft_volume_2, slice_idx=80)
```

### 3. Histogram Analysis

```python
plot_grayscale_histogram(fft_volume)
plot_image_histograms("path/to/image1.png", "path/to/image2.png")
```

### 4. Normalized Cross-Correlation (NCC)

```python
ncc_value = normalized_cross_correlation(image1_array, image2_array)
```

## Batch Evaluation Example (Commented in Script)

- Batch PSNR, SSIM, and NCC comparisons for full image sets (e.g., axial or sagittal slices).
- Uncomment corresponding blocks to activate.

## Error Map Visualization (Optional)

Generate absolute error maps and histogram overlays between low-res and high-res images:

```python
error_map = np.abs(hr_np - lr_np)
plt.imshow(error_map, cmap='hot')
plt.title("Error Map")
```
