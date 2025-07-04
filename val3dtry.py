import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

FUSED_DIR = 'fused_minmax_volumes'
GT_DIR = 'axial_volumes_npy/ground_truth'

def calculate_psnr(original, compared, max_val=255.0):
    mse = np.mean((original - compared) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)

def evaluate_all(fused_dir, gt_dir):
    fused_files = sorted([f for f in os.listdir(fused_dir) if f.endswith('.npy')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

    common_files = set(fused_files).intersection(set(gt_files))
    if not common_files:
        print("No matching files found between fused and GT directories!")
        return

    psnr_scores = []
    ssim_scores = []

    for filename in sorted(common_files):
        fused_vol = np.load(os.path.join(fused_dir, filename)).astype(np.float32)
        gt_vol = np.load(os.path.join(gt_dir, filename)).astype(np.float32)

        if fused_vol.shape != gt_vol.shape:
            from skimage.transform import resize
            fused_vol = resize(fused_vol, gt_vol.shape, preserve_range=True, anti_aliasing=True).astype(np.float32)

        if np.std(gt_vol) == 0 or np.std(fused_vol) == 0:
            print(f"Skipping {filename} due to zero variance in data.")
            continue

        max_val = max(gt_vol.max(), fused_vol.max())
        psnr_val = calculate_psnr(gt_vol, fused_vol, max_val=max_val)
        ssim_val = np.mean([ssim(gt_vol[i], fused_vol[i], data_range=gt_vol[i].max() - gt_vol[i].min()) for i in range(gt_vol.shape[0])])

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        print(f"{filename}: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}")

    print(f"\nAverage PSNR: {np.mean(psnr_scores):.2f}")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")

if __name__ == "__main__":
    evaluate_all(FUSED_DIR, GT_DIR)
