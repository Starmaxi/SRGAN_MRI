import os
import numpy as np
from pathlib import Path
from PIL import Image

main_folder = Path("C:/Users/LaraR/OneDrive/Desktop/Projekt_MRI/sorted_val_2/val_processed/coronal") #anpassen an ordner

output_folder = os.path.join(main_folder, "volumes")
os.makedirs(output_folder, exist_ok=True)

#Numerisch sortieren nach Dateinamen
def numeric_key(filename):
    return int(os.path.splitext(filename)[0])

for subfolder_name in sorted(os.listdir(main_folder)):
    subfolder_path = os.path.join(main_folder, subfolder_name)

    if os.path.isdir(subfolder_path):
        image_files = sorted([
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ], key=numeric_key)

        if not image_files:
            print(f"Keine Bilder in {subfolder_name}")
            continue

        image_stack = []
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            with Image.open(image_path) as img:
                img = img.convert("L") 
                img_array = np.array(img)
                image_stack.append(img_array)

        volume = np.stack(image_stack, axis=0)
        output_path = os.path.join(output_folder, f"{subfolder_name}.npy")
        np.save(output_path, volume)

        print(f"{output_path} gespeichert | Form: {volume.shape}")


