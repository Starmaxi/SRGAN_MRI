'''
Ausgang: 3 hauptordner mit je 1 unterordner mit 10 volumen
Ziel: 1 hauptordner mit 10 neuen volumen
mittelwert aus volumen berechnen

'''
from pathlib import Path
import numpy as np

# 3 Hauptordner, jeweils mit einem Unterordner "volumes" (enthalten gehirn1.npy bis gehirn10.npy)
hauptordner_liste = [
    Path("C:/Users/LaraR/OneDrive/Desktop/Projekt_MRI/sorted_val_2/val_processed/sagittal"),
    Path("C:/Users/LaraR/OneDrive/Desktop/Projekt_MRI/sorted_val_2/val_processed/coronal"),
    Path("C:/Users/LaraR/OneDrive/Desktop/Projekt_MRI/sorted_val_2/val_processed/axial")
]

# Zielordner für gemittelte Volumen
zielordner = Path("C:/Users/LaraR/OneDrive/Desktop/Projekt_MRI/sorted_val_2/val_processed")
zielordner.mkdir(parents=True, exist_ok=True)

# Liste der Volumendateien: gehirn1.npy bis gehirn10.npy
dateinamen = [f"gehirn{i}.npy" for i in range(1, 11)]

for dateiname in dateinamen:
    volumen_liste = []

    for hauptordner in hauptordner_liste:
        volumenpfad = hauptordner / "volumes" / dateiname

        if not volumenpfad.is_file():
            raise FileNotFoundError(f"{volumenpfad} wurde nicht gefunden.")
        
        volumen = np.load(volumenpfad)
        volumen_liste.append(volumen)

    shapes = [v.shape for v in volumen_liste]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"Volumenformen für {dateiname} stimmen nicht überein: {shapes}")

    gemitteltes_volumen = np.mean(volumen_liste, axis=0)

    speicherpfad = zielordner / dateiname
    np.save(speicherpfad, gemitteltes_volumen)
    print(f"{dateiname} gespeichert nach {speicherpfad}")


