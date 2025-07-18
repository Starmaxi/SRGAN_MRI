 
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from datasets import *
#import cv2
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vgg19
import torch.optim as optim
from models import *
from tqdm import tqdm
import torch.multiprocessing as mp
#torch.multiprocessing.set_sharing_strategy('file_system')

def sobel_edges(x):
    # x: [B, H, W], grayscale
    B, H, W = x.shape
    x = x.view(B, 1, H, W)

    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=x.dtype, device=x.device)

    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]]]], dtype=x.dtype, device=x.device)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    kernel_size = 8
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    padding = kernel_size // 2


    edge_magnitude = F.max_pool2d(x,kernel_size=16, stride=1, padding=8)
    grad_x = F.conv2d(edge_magnitude, sobel_x, padding=1)
    grad_y = F.conv2d(edge_magnitude, sobel_y, padding=1)
    edge_magnitude = (torch.sqrt(grad_x**2 + grad_y**2))
    edge_magnitude = F.sigmoid(F.max_pool2d(edge_magnitude,kernel_size=8, stride=1, padding=4))
    edge_magnitude = torch.where(edge_magnitude < edge_magnitude.mean(), torch.tensor(0, dtype=torch.uint8), torch.tensor(1, dtype=torch.float32))

    return edge_magnitude.squeeze(1)  # Shape: [B, H, W]


def draw_bounding_box(img, tensor_input, image_size=256):
    """
    Zeichnet eine Bounding Box in einen 256x256-Tensor basierend auf [x, y, r].

    Args:
        tensor_input (torch.Tensor): Eingabe-Tensor der Form [x, y, r].
        image_size (int): Größe des Ausgabe-Tensors (default: 256).

    Returns:
        torch.Tensor: Ein Binär-Tensor der Größe (image_size, image_size),
                      wo 1 die Bounding Box markiert und 0 der Hintergrund ist.
    """
    # Extrahiere x, y, r aus dem Eingabe-Tensor
    x_center, y_center, r_x, r_y = tensor_input

    # Erstelle ein leeres Gitter (256x256)

    # Berechne die Koordinaten der Bounding-Box-Ecken
    x_min = int(x_center - r_x)
    x_max = int(x_center + r_x)
    y_min = int(y_center - r_y)
    y_max = int(y_center + r_y)

    # Stelle sicher, dass die Koordinaten innerhalb des Bildes liegen (0-255)
    x_min = max(0, x_min)
    x_max = min(image_size - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(image_size - 1, y_max)

    # Zeichne die Bounding Box (setze Pixel auf 1)

    img[y_min:y_max+1, x_min] = 1  # Linke Kante
    img[y_min:y_max+1, x_max] = 1  # Rechte Kante
    img[y_min, x_min:x_max+1] = 1  # Obere Kante
    img[y_max, x_min:x_max+1] = 1  # Untere Kante

    return img

def process_batch(i, imgs, model, normalize, out_path):
    sobel_output = sobel_edges(imgs["hr"][0])[0]
    pred = model(normalize(sobel_output.unsqueeze(0).unsqueeze(0)))
    torch.save({"hr": imgs["hr"].squeeze(0).to(torch.float16).detach(), "bb": pred.to(torch.float16).detach()}, out_path + "tensors_" + str(i) + ".pt")


if __name__ == "__main__":
    out_path = "/mnt/1tb_ssd/Dokumente/Schule/THI/projekt_2/tensor_data/all_views_bb_brain_only/"

    hr_shape = [256,256]
    batch_size = 1
    n_cpu = 16

    # DataLoader
    dataloader = DataLoader(
        ImageDataset("/mnt/1tb_ssd/Dokumente/Schule/THI/projekt_2/image_data/brain_only/joint_data/", hr_shape=hr_shape),
        #ImageDataset("folder_with_images/", hr_shape=hr_shape),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=False
    )

    model = CoordRegressionNet()
    model.load_state_dict(torch.load("model_4_outputs.pth.best"))
    model.eval()

    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    with mp.Pool(processes=n_cpu) as pool:
        with tqdm(total = len(dataloader)) as pbar:
            for i, imgs in enumerate(dataloader):
                #pool.apply_async(process_batch, args=(imgs, model, normalize, out_path))
                pool.apply(process_batch, args=(i, imgs, model, normalize, out_path))
                pbar.update(1)

