#uncertainty map berechnen
#Monte Carlo Dropout

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

def sr_with_uncertainty(generator, lr_image, n_iter=30):
    generator.train()  # Dropout bleibt aktiv
    sr_images = [generator(lr_image) for _ in range(n_iter)]
    sr_stack = torch.stack(sr_images)  # [n_iter, 1, H, W]
    mean_sr = sr_stack.mean(dim=0) #fertige SR Bild (mittel der 30 durchläufe)
    var_map = sr_stack.var(dim=0) #uncertainty map
    return mean_sr, var_map

def save_uncertainty_map(var_map, save_path):
    std_map = var_map.mean(dim=0).sqrt().cpu().numpy()  # [H, W]
    plt.imsave(save_path, std_map, cmap='inferno')

def process_folder(generator, input_dir, output_dir, n_iter=30, device='cuda'):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

    generator.to(device)

    for img_path in tqdm(sorted(input_dir.glob("*.png")), desc="Processing images"):
        lr_image = Image.open(img_path).convert("L")
        lr_tensor = transform(lr_image).unsqueeze(0).to(device) 

        with torch.no_grad():
            sr_img, var_map = sr_with_uncertainty(generator, lr_tensor, n_iter)

        #Bild speichern
        sr_img_np = sr_img.squeeze().clamp(0, 1).cpu().numpy() * 255
        sr_img_pil = Image.fromarray(sr_img_np.astype(np.uint8))
        sr_save_path = output_dir / f"{img_path.stem}_sr.png"
        sr_img_pil.save(sr_save_path)

        #Map speichern
        unc_save_path = output_dir / f"{img_path.stem}_uncertainty.png"
        save_uncertainty_map(var_map, unc_save_path)

#funktion process folder aufrufen mit outputfolder name, input folder name und generator
