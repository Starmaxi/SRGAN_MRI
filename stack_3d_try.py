import os
import numpy as np
from PIL import Image

def load_axial_volume(folder_path, target_size=None):
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])  # assumes slices are named "0.png", "1.png", etc.
    )
    volume = []
    for fname in files:
        img = Image.open(os.path.join(folder_path, fname)).convert('L')  # grayscale
        img_np = np.array(img)
        if target_size:
            img_np = np.array(Image.fromarray(img_np).resize(target_size, Image.BILINEAR))
        volume.append(img_np)
    return np.stack(volume, axis=0)

def process_axial_dataset(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subject_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # Step 1: Find smallest shape
    min_w, min_h = np.inf, np.inf
    for subject in subject_folders:
        subject_path = os.path.join(base_dir, subject)
        files = sorted([f for f in os.listdir(subject_path) if f.endswith('.png')])
        if not files:
            continue
        img = Image.open(os.path.join(subject_path, files[0])).convert('L')
        h, w = np.array(img).shape
        min_w, min_h = min(min_w, w), min(min_h, h)
    target_size = (int(min_w), int(min_h))

    for subject in subject_folders:
        try:
            subject_path = os.path.join(base_dir, subject)
            volume = load_axial_volume(subject_path, target_size)
            save_path = os.path.join(output_dir, f"{subject}.npy")
            np.save(save_path, volume)
            print(f"Saved {save_path}, shape: {volume.shape}")
        except Exception as e:
            print(f"Failed on {subject}: {e}")

process_axial_dataset(
    base_dir='brain_result_val/ground_truth/axial',
    output_dir='axial_volumes_npy/ground_truth'
)

process_axial_dataset(
    base_dir='brain_result_val/high_resolution/axial',
    output_dir='axial_volumes_npy/high_resolution'
)
