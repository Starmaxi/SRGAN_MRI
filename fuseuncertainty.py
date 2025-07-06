import numpy as np
from pathlib import Path
import os


input_dir = Path("C:/Users/LaraR/OneDrive/Desktop/Projekt_MRI/sorted_val_2/val_processed/sagittal/uncertainty") #anpassen axial, coronal
output_dir = input_dir.parent / "fusion_output"
output_dir.mkdir(exist_ok=True)

all_sr_files = sorted(list(input_dir.glob("*_sr.npy")))
gehirn_names = sorted(set(f.name.split("_")[0] for f in all_sr_files))

for gehirn in gehirn_names:
    sr_paths = sorted(input_dir.glob(f"{gehirn}_sr*.npy"))
    unc_paths = [Path(str(p).replace("_sr", "_uncertainty")) for p in sr_paths]

    assert len(sr_paths) == 3 and all(p.exists() for p in unc_paths), f"{gehirn}: Unvollständige Dateien"

    sr_volumes = [np.load(p) for p in sr_paths]
    unc_volumes = [np.load(p) for p in unc_paths]

    sr_stack = np.stack(sr_volumes, axis=0)
    unc_stack = np.stack(unc_volumes, axis=0)

    min_indices = np.argmin(unc_stack, axis=0)
    fused_volume = np.zeros_like(sr_volumes[0])

    D, H, W = fused_volume.shape
    for z in range(D):
        for y in range(H):
            for x in range(W):
                fused_volume[z, y, x] = sr_stack[min_indices[z, y, x], z, y, x]

    output_path = output_dir / f"{gehirn}_fused.npy"
    np.save(output_path, fused_volume)
    print(f"{gehirn}: gespeichert unter {output_path}")
