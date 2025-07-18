# 🧠 MRI Super-Resolution via SR-GAN

Dieses Projekt enthält Skripte zur Verarbeitung und Fusion von MRT-Bildern mit Super-Resolution und Unsicherheitsbewertung.

## 📜 Skripte

1. `fuseuncertainty.py` – Unsicherheitsbasierte voxelweise Fusion  
2. *(weitere Skripte folgen, z. B. `preprocess.py`, `evaluate.py`, `visualize.py`)*

---

## 🔧 `fuseuncertainty.py`

### Zweck  
Führt eine voxelweise Fusion von drei rekonstruierten MRT-Volumes (z. B. axial, koronal, sagittal) durch und wählt pro Voxel den Wert mit der geringsten Unsicherheit.

### Eingabe  
Ordner mit Dateien wie:

- `001_sr0.npy`, `001_sr1.npy`, `001_sr2.npy`  
- `001_uncertainty0.npy`, `001_uncertainty1.npy`, `001_uncertainty2.npy`

### Ausgabe  
- `001_fused.npy` im Ordner `fusion_output/`

### Nutzung

```bash
python fuseuncertainty.py
