import glob
import random
import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Normalization parameters for pre-trained PyTorch models
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
mean=np.array([0.5])
std=np.array([0.5])

def fill_between_ovals_with_noise(tensor, inner_center, inner_radius, outer_center, outer_radius, noise_mean=0, noise_std=1):
    """
    Füllt den Zwischenraum zwischen zwei Ovalen mit Rauschen.

    Args:
        tensor (torch.Tensor): Ein 256x256 Tensor.
        inner_center (tuple): (x, y) Koordinaten des Mittelpunkts des inneren Ovals.
        inner_radius (tuple): (rx, ry) Radien des inneren Ovals in x- und y-Richtung.
        outer_center (tuple): (x, y) Koordinaten des Mittelpunkts des äußeren Ovals.
        outer_radius (tuple): (rx, ry) Radien des äußeren Ovals in x- und y-Richtung.
        noise_mean (float): Mittelwert des Rauschens.
        noise_std (float): Standardabweichung des Rauschens.

    Returns:
        torch.Tensor: Tensor mit Rauschen im Zwischenraum der Ovale.
    """
    # Erstelle ein Gitter von Koordinaten
    x = torch.arange(256).float()
    y = torch.arange(256).float()
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Koordinaten relativ zu den Mittelpunkten setzen
    #inner_xx = (xx - inner_center[0]) / inner_radius[0]
    #inner_yy = (yy - inner_center[1]) / inner_radius[1]
    #outer_xx = (xx - outer_center[0]) / outer_radius[0]
    #outer_yy = (yy - outer_center[1]) / outer_radius[1]

    inner_xx = (xx - inner_center[1]) / inner_radius[1]
    inner_yy = (yy - inner_center[0]) / inner_radius[0]
    outer_xx = (xx - outer_center[1]) / outer_radius[1]
    outer_yy = (yy - outer_center[0]) / outer_radius[0]

    # Berechne die Ellipsengleichungen
    inner_oval = (inner_xx ** 2 + inner_yy ** 2) >= 1
    outer_oval = (outer_xx ** 2 + outer_yy ** 2) <= 1

    # Maske für den Bereich zwischen den Ovalen
    between_mask = torch.logical_and(inner_oval, outer_oval)

    # Generiere Rauschen
    #noise = torch.normal(mean=noise_mean, std=noise_std, size=tensor.shape)
    noise = block_noise(torch.rand(1,256,256))
    #print(noise.size())
    #plt.imshow(noise.squeeze(0), cmap='gray')
    #plt.show()
    # Fülle den Zwischenraum mit Rauschen
    tensor[between_mask] = noise[between_mask]

    return tensor

def block_noise(x):
    # x: [B, H, W], grayscale
    #B, H, W = x.shape
    #x = x.view(B, 1, H, W)

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


    #edge_magnitude = F.max_pool2d(x,kernel_size=16, stride=1, padding=8)
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    edge_magnitude = (torch.sqrt(grad_x**2 + grad_y**2))
    edge_magnitude = F.max_pool2d(edge_magnitude,kernel_size=16, stride=1, padding=8)
    edge_magnitude = torch.where(edge_magnitude < edge_magnitude.mean(), torch.tensor(0, dtype=torch.uint8), torch.tensor(1, dtype=torch.float16))

    return edge_magnitude.squeeze(0)[:256, :256]  # Shape: [B, H, W]


def add_geometric_binary_noise(input_tensor, size):
    """
    Fügt binäres Rauschen (0 oder 1) zu einem zufälligen geometrischen Patch in einem 256x256 Tensor hinzu.

    Args:
        input_tensor (torch.Tensor): Eingabetensor der Größe 256x256

    Returns:
        torch.Tensor: Tensor mit verrauschtem Patch
    """
    assert input_tensor.shape == (256, 256), "Input tensor muss 256x256 sein"

    # Kopie des Tensors erstellen
    #noisy_tensor = input_tensor.clone()
    noisy_tensor = input_tensor

    # Zufällige geometrische Form auswählen
    shape_type = random.choice(['rectangle', 'circle', 'triangle', 'ellipse'])

    # Zufällige Position und Größe
    center_x = random.randint(50, 206)
    center_y = random.randint(50, 206)
    #size = random.randint(20, 100)

    # Binäres Rauschen erzeugen
    binary_noise = (torch.rand(size, size) > 0.5).float()

    # Patch entsprechend der gewählten Form einfügen
    if shape_type == 'rectangle':
        # Zufällige Rechteck-Dimensionen
        width = random.randint(10, size)
        height = random.randint(10, size)
        start_x = max(0, center_x - width//2)
        start_y = max(0, center_y - height//2)
        end_x = min(256, start_x + width)
        end_y = min(256, start_y + height)

        # Rauschen auf rechteckigen Bereich anwenden
        noisy_tensor[start_y:end_y, start_x:end_x] = binary_noise[:end_y-start_y, :end_x-start_x]

    elif shape_type == 'circle':
        # Kreis-Patch
        for y in range(max(0, center_y-size), min(256, center_y+size)):
            for x in range(max(0, center_x-size), min(256, center_x+size)):
                if math.sqrt((x-center_x)**2 + (y-center_y)**2) <= size:
                    noisy_tensor[y, x] = 1 if random.random() > 0.5 else 0

    elif shape_type == 'ellipse':
        # Ellipse mit zufälligen Radien
        radius_x = random.randint(10, size)
        radius_y = random.randint(10, size)
        for y in range(max(0, center_y-radius_y), min(256, center_y+radius_y)):
            for x in range(max(0, center_x-radius_x), min(256, center_x+radius_x)):
                if ((x-center_x)/radius_x)**2 + ((y-center_y)/radius_y)**2 <= 1:
                    noisy_tensor[y, x] = 1 if random.random() > 0.5 else 0

    elif shape_type == 'triangle':
        # Zufälliges Dreieck
        points = [
            (center_x, center_y - size),
            (center_x - size, center_y + size),
            (center_x + size, center_y + size)
        ]
        # Dreieck füllen
        for y in range(max(0, center_y-size), min(256, center_y+size)):
            for x in range(max(0, center_x-size), min(256, center_x+size)):
                if point_in_triangle((x,y), points):
                    noisy_tensor[y, x] = 1 if random.random() > 0.5 else 0

    #return noisy_tensor

def point_in_triangle(point, triangle):
    """Hilfsfunktion um zu prüfen ob ein Punkt in einem Dreieck liegt"""
    x, y = point
    (x1, y1), (x2, y2), (x3, y3) = triangle

    def cross_product(a, b):
        return a[0]*b[1] - a[1]*b[0]

    A = cross_product((x2-x1, y2-y1), (x3-x1, y3-y1))
    A1 = cross_product((x2-x, y2-y), (x3-x, y3-y))
    A2 = cross_product((x3-x, y3-y), (x1-x, y1-y))
    A3 = cross_product((x1-x, y1-y), (x2-x, y2-y))

    return (A1 >= 0 and A2 >= 0 and A3 >= 0) if A >= 0 else (A1 <= 0 and A2 <= 0 and A3 <= 0)


def draw_circle_line(tensor: torch.Tensor, center_x: int, center_y: int, radius: int, thickness: int = 1):
    """
    Zeichnet einen Kreis als Linie (1 Pixel breit) in den Tensor.
    Args:
        tensor: 2D-Tensor (Höhe x Breite).
        center_x, center_y: Mittelpunkt des Kreises.
        radius: Radius des Kreises.
        thickness: Linienstärke (Standard: 1 Pixel).
    """
    h, w = tensor.shape
    for y in range(h):
        for x in range(w):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if abs(distance - radius) <= thickness / 2:
                tensor[y, x] = 1

def draw_oval_line(tensor: torch.Tensor, center_x: int, center_y: int, radius_x: int, radius_y: int, thickness: int = 1):
    """
    Zeichnet ein Oval als Linie (1 Pixel breit) in den Tensor.
    Args:
        tensor: 2D-Tensor (Höhe x Breite).
        center_x, center_y: Mittelpunkt des Ovals.
        radius_x: Horizontaler Radius.
        radius_y: Vertikaler Radius.
        thickness: Linienstärke (Standard: 1 Pixel).
    """
    h, w = tensor.shape
    for y in range(h):
        for x in range(w):
            # Oval-Formel: ((x - cx)/rx)² + ((y - cy)/ry)² ≈ 1
            distance = math.sqrt(((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2)
            if abs(distance - 1) <= thickness / (2 * max(radius_x, radius_y)):
                tensor[y, x] = 1

def draw_random_shape(tensor, center_x, center_y, min_radius, radius, thickness, y_rad_mul=1):
    h, w = tensor.shape
    """
    if random.random() > 0.8:  # 50% Chance für Kreis oder Oval
        draw_circle_line(tensor, center_x, center_y, radius, thickness)

        return center_x, center_y, radius, radius
    """
    #else:
    if int(radius*y_rad_mul) <= min_radius:
        radius_y = min_radius
    else:
        radius_y = random.randint(min_radius, int(radius*y_rad_mul))

    if random.random() > 0.5:
        draw_oval_line(tensor, center_x, center_y, radius, radius_y, thickness)
        return center_x, center_y, radius, radius_y
    else:
        draw_oval_line(tensor, center_x, center_y, radius_y, radius, thickness)
        return center_x, center_y, radius_y, radius

    #draw_oval_line(tensor, center_x, center_y, radius_y, radius, thickness)
    #return center_x, center_y, radius, radius_y

def draw_inner_shape(tensor: torch.Tensor, max_radius: int = 20, min_radius: int = 20, thickness: int = 1, outer_center_x = 0, outer_center_y = 0):
    radius = random.randint(min_radius, max_radius-1)
    center_x = random.randint(outer_center_x-max_radius+radius, outer_center_x + max_radius -radius - 1)
    center_y = random.randint(outer_center_y-max_radius+radius, outer_center_y + max_radius -radius - 1)
    return draw_random_shape(tensor, center_x, center_y, min_radius, radius, thickness, 0.75)

def draw_outer_shape(tensor: torch.Tensor, max_radius: int = 20, min_radius: int = 20, thickness: int = 1):
    h, w = tensor.shape
    radius = random.randint(min_radius, max_radius)
    center_x = random.randint(radius, w - radius - 1)
    center_y = random.randint(radius, h- radius - 1)
    return draw_random_shape(tensor, center_x, center_y, min_radius, radius, thickness)

def add_random_noise(tensor: torch.Tensor,
    noise_density: float = 0.01,
    max_strength: int = 3,
    min_strength: int = 3,
    noise_value: int = 1
):
    """
    Fügt Rauschen (Wert=1) mit variabler Stärke (Blockgröße) hinzu.
    Args:
        tensor: 2D-Tensor (Höhe x Breite).
        noise_density: Anteil der Rauschpixel (relativ zur Tensor-Größe).
        max_strength: Maximale Größe der Rauschblöcke (z. B. 3 = 3x3-Pixel-Blöcke).
    """
    h, w = tensor.shape
    num_noise_pixels = int(noise_density * h * w)

    for _ in range(num_noise_pixels):
        # Zufällige (y, x)-Position
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        # Zufällige Stärke (1 = einzelner Pixel, max_strength = größerer Block)
        strength = random.randint(min_strength, max_strength)

        # Setze einen Block der Größe `strength x strength` auf 1
        for dy in range(strength):
            for dx in range(strength):
                if y + dy < h and x + dx < w:  # Randüberprüfung
                    tensor[y + dy, x + dx] = noise_value

    return tensor

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

class TensorBBDataset(Dataset):
    def __init__(self, root, hr_height):#
        self.hr_height = hr_height
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        input_data = torch.load(self.files[index % len(self.files)])
        img_hr = input_data["hr"]

        return {"img": img_hr.squeeze(0), "bb": input_data["bb"].squeeze(0).squeeze(0)}

    def __len__(self):
        return len(self.files)

def collate_fn(batch):
    img_tensors = [item["img"] for item in batch]
    batch_bboxes = torch.stack([item["bb"] for item in batch])*256

    # find biggest bb
    rad_x = int(batch_bboxes[:,2].max())
    rad_y = int(batch_bboxes[:,3].max())

    cropped_tensors = []

    for tensor, bb in zip(img_tensors, batch_bboxes):
        #print("start")
        center_x = int(bb[0])
        center_y = int(bb[1])
        pad_neg = 0
        pad_pos = 0
        #print(tensor.shape)
        #print("rad: ", rad_x, rad_y)
        #print("new size y: ", center_y+rad_y-(center_y-rad_y))
        #print("new size x: ", center_x+rad_x-(center_x-rad_x))
        #print("center_x + rad_x: ", center_x+rad_x)
        #print("center_x - rad_x: ", center_x-rad_x)
        #print("center_y + rad_y: ", center_y+rad_y)
        #print("center_y - rad_y: ", center_y-rad_y)
        #print("center_x:", center_x)
        #print("center_y:", center_y)

        if (rad_x) // 4 != 0 or (rad_y) // 4 != 0:
            rad_x = math.ceil(rad_x // 4) * 4
            rad_y = math.ceil(rad_y // 4) * 4

        x_max = center_x+rad_x
        x_min = center_x-rad_x
        y_max = center_y+rad_y
        y_min = center_y-rad_y

        if torch.any(torch.tensor([x_max, x_min, y_max, y_min]) < 0):
            pad_neg = abs(int(torch.tensor([x_max, x_min, y_max, y_min]).min()))
            #print("pad:", pad_neg)
            tensor = F.pad(tensor, (pad_neg, pad_neg,pad_neg, pad_neg), mode="constant", value = 0)
            #print("after padding:", tensor.shape)
        if torch.any(torch.tensor([x_max, x_min, y_max, y_min]) > 256):
            pad_pos = int(torch.tensor([x_max, x_min, y_max, y_min]).max()) - 256
            #print("pad:", pad_pos)
            tensor = F.pad(tensor, (pad_pos, pad_pos,pad_pos, pad_pos), mode="constant", value = 0)
            #print("after padding:", tensor.shape)

        #plt.imshow(tensor[center_y+pad-rad_y:center_y+pad+rad_y,center_x+pad-rad_x:center_x+pad+rad_x])
        #plt.show()
        #print(tensor[center_y+pad-rad_y:center_y+pad+rad_y,center_x+pad-rad_x:center_x+pad+rad_x].shape)
        #cropped_tensors.append(tensor[center_y+pad_pos+pad_neg-rad_y:center_y+pad_pos+pad_neg+rad_y,center_x+pad_pos+pad_neg-rad_x:center_x+pad_pos+pad_neg+rad_x].unsqueeze(0))
        cropped_tensors.append(tensor[y_min+pad_pos+pad_neg:y_max+pad_pos+pad_neg,x_min+pad_pos+pad_neg:x_max+pad_pos+pad_neg].unsqueeze(0))
    cropped_tensors = torch.stack(cropped_tensors)
    #return cropped_tensors
    return {"lr": F.interpolate(cropped_tensors,size=(cropped_tensors.shape[2]//4, cropped_tensors.shape[3]//4), mode='bicubic', align_corners=False), "hr": cropped_tensors}

class CircleDataset(Dataset):
    def __init__(self, shape, data_len):
        self.shape = shape
        self.data_len = data_len
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __getitem__(self, index):
        input_tensor = torch.zeros((self.shape, self.shape)).to(torch.float16)  # Leerer Tensor (Höhe x Breite)

        # Zufällige Kreise/Ovale zeichnen
        noise_density = random.uniform(0.01,0.06)
        thickness = random.randint(8,16)
        # draw outer circle
        center_x_o, center_y_o, radius_x_o, radius_y_o = draw_outer_shape(input_tensor, max_radius=127, min_radius=71, thickness=thickness)
        # draw inner circle with 0.7 probability
        if random.random() > 0.3:
            thickness = random.randint(8,16)
            center_x, center_y, radius_x, radius_y = draw_inner_shape(input_tensor, max_radius=min(radius_x_o, radius_y_o), min_radius=50, thickness=thickness, outer_center_x=center_x_o, outer_center_y=center_y_o)
            input_tensor = fill_between_ovals_with_noise(input_tensor, (center_x, center_y), (radius_x, radius_y), (center_x_o, center_y_o), (radius_x_o, radius_y_o))
            input_tensor = add_random_noise(input_tensor, min_strength = 3, noise_value = 0,noise_density=noise_density)
            if radius_x > radius_y:
                radius_x += 5
            else:
                radius_y += 5
            return {"img": self.normalize(input_tensor.unsqueeze(0)), "y": torch.tensor([center_x, center_y, radius_x, radius_y])/self.shape}

        if radius_x_o > radius_y_o:
            radius_x_o += 5
        else:
            radius_y_o += 5
        input_tensor = add_random_noise(input_tensor, min_strength = 3, noise_value = 0,noise_density=noise_density)
        return {"img": self.normalize(input_tensor.unsqueeze(0)), "y": torch.tensor([center_x_o, center_y_o, radius_x_o, radius_y_o])/self.shape}

    def __len__(self):
        return self.data_len
