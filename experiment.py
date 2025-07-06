import numpy as np
import os

def min_max_fusion(vol1, vol2, mode='max'):
    assert vol1.shape == vol2.shape, "Volumes must have the same shape"
    if mode == 'min':
        return np.minimum(vol1, vol2)
    elif mode == 'max':
        return np.maximum(vol1, vol2)
    else:
        raise ValueError("Mode must be 'min' or 'max'")

highres_dir = 'axial_volumes_npy/high_resolution'
gt_dir = 'axial_volumes_npy/ground_truth'
output_dir = 'fused_minmax_volumes'
os.makedirs(output_dir, exist_ok=True)

subjects = [f for f in os.listdir(highres_dir) if f.endswith('.npy')]

for subj in subjects:
    vol_hr_path = os.path.join(highres_dir, subj)
    vol_gt_path = os.path.join(gt_dir, subj)

    if not os.path.exists(vol_gt_path):
        continue

    vol_hr = np.load(vol_hr_path)
    vol_gt = np.load(vol_gt_path)

    fused_max = min_max_fusion(vol_hr, vol_gt, mode='max')
    fused_min = min_max_fusion(vol_hr, vol_gt, mode='min')

    np.save(os.path.join(output_dir, subj.replace('.npy', '_max_fused.npy')), fused_max)

    print(f"Fused volumes saved for {subj}")

